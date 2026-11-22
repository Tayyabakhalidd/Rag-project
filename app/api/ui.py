"""Server-rendered HTML pages (Jinja2 templates + Pico.css, no JavaScript).

The page uses ordinary HTML forms. Each form posts to a UI route that calls
the *same* services as the JSON API, so the browser and the API behave the
same way.

* Success after an upload or delete -> redirect to ``/`` with a short notice
  (Post/Redirect/Get, so refreshing the page does not resubmit the form).
* Errors -> the page is rendered again with the error banner and the real
  HTTP status code (for example 409 for a duplicate).
* A question -> the page is rendered with the answer, its sources and the
  retrieved chunks with their similarity scores.

These routes are hidden from the OpenAPI docs.
"""

import re
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.api.deps import get_document_service, get_qa_service, limit_questions, limit_uploads
from app.config import MAX_QUESTION_CHARS, MAX_TOP_K, PROJECT_ROOT
from app.errors import AppError
from app.ingestion.validation import read_upload_limited
from app.schemas import AskRequest

FRONTEND_DIR = PROJECT_ROOT / "frontend"
templates = Jinja2Templates(directory=str(FRONTEND_DIR))  # autoescape is on for .html files

router = APIRouter(include_in_schema=False)

STATUS_TITLES = {
    400: "Bad request",
    404: "Not found",
    409: "Duplicate",
    413: "Too large",
    415: "Unsupported file type",
    422: "Invalid input",
    429: "Too many requests",
    503: "Service unavailable",
}


def cite_links(text: str) -> Markup:
    """Escape the answer, then turn [C2] markers into links to the matching chunk card."""
    escaped = str(escape(text))
    return Markup(re.sub(r"\[(C\d+)\]", r'<a class="cite" href="#chunk-\1">[\1]</a>', escaped))


templates.env.filters["cite_links"] = cite_links


def _error_view(exc: AppError) -> dict[str, Any]:
    details = dict(exc.details)
    if exc.code == "duplicate_document" and "filename" in details:
        message = f"{exc.message} It is already in the list as {details['filename']}."
    else:
        message = exc.message
    return {"status": exc.status_code, "title": STATUS_TITLES.get(exc.status_code, "Error"), "code": exc.code, "message": message}


def _render(request: Request, status_code: int = 200, **context: Any) -> HTMLResponse:
    settings = request.app.state.settings
    documents = get_document_service(request).list_documents()
    base = {
        "documents": documents,
        "notice": None,
        "error": None,
        "question": "",
        "top_k": settings.top_k,
        "result": None,
        "max_question_chars": MAX_QUESTION_CHARS,
        "max_top_k": MAX_TOP_K,
        "llm_configured": settings.groq_configured,
        "embeddings_configured": settings.gemini_configured,
    }
    base.update(context)
    return templates.TemplateResponse(request, "index.html", base, status_code=status_code)


def _redirect(**params: str) -> RedirectResponse:
    return RedirectResponse("/?" + urlencode(params), status_code=303)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    notice = None
    service = get_document_service(request)
    action = request.query_params.get("notice")
    if action == "uploaded":
        document = service.registry.get(request.query_params.get("doc", ""))
        if document:
            notice = f"Uploaded {document['filename']}: {document['chunk_count']} chunks indexed."
    elif action == "deleted":
        notice = f"Deleted {request.query_params.get('name', 'the document')} and all of its chunks."
    return _render(request, notice=notice)


@router.get("/static/styles.css")
async def styles() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@router.post("/ui/documents", response_class=HTMLResponse)
async def upload(request: Request) -> Response:
    try:
        limit_uploads(request)
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile) or not file.filename:
            return _render(
                request, 422, error={"status": 422, "title": "Invalid input", "code": "validation_error",
                                     "message": "Choose a PDF or DOCX file to upload."}
            )
        data = await read_upload_limited(file, request.app.state.settings.max_upload_bytes)
        document = await get_document_service(request).upload(file.filename, data)
    except AppError as exc:
        return _render(request, exc.status_code, error=_error_view(exc))
    return _redirect(notice="uploaded", doc=document["document_id"])


@router.post("/ui/documents/{document_id}/delete", response_class=HTMLResponse)
async def delete(request: Request, document_id: str) -> Response:
    try:
        document = await get_document_service(request).delete(document_id)
    except AppError as exc:
        return _render(request, exc.status_code, error=_error_view(exc))
    return _redirect(notice="deleted", name=document["filename"])


@router.post("/ui/ask", response_class=HTMLResponse)
async def ask(request: Request) -> HTMLResponse:
    form = await request.form()
    question = str(form.get("question", ""))
    top_k_raw = str(form.get("top_k", "")).strip()
    try:
        body = AskRequest(question=question, top_k=int(top_k_raw) if top_k_raw else None)
    except (ValidationError, ValueError):
        return _render(
            request, 422, question=question,
            error={"status": 422, "title": "Invalid input", "code": "validation_error",
                   "message": f"Enter a question of 1 to {MAX_QUESTION_CHARS} characters and a top-k of 1 to {MAX_TOP_K}."},
        )
    try:
        limit_questions(request)
        result = await get_qa_service(request).ask(body.question, body.top_k)
    except AppError as exc:
        return _render(request, exc.status_code, question=question, top_k=body.top_k, error=_error_view(exc))
    return _render(request, question=body.question, top_k=body.top_k or request.app.state.settings.top_k, result=result)
