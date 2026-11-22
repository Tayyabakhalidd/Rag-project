# AI usage

The course allows AI tools as learning and development assistants, provided their use is documented. This file records how AI was used in this project.

## Tools and models

| Tool | Model | Used for |
|---|---|---|
| Claude Code (VS Code extension, agent mode) | Claude Opus 5.5 (`claude-opus-5-5`) | Planning, implementation, tests, the evaluation code, the notebook, documentation |
| claude.ai artifact | Claude | The written project plan (the "Production RAG Service Plan" page, kept outside the repository) |

Models used *inside* the product, not as assistants: a Groq-hosted LLM for answers, a different Groq-hosted LLM as the evaluation judge, and Gemini embeddings. Their model IDs are recorded in each evaluation summary (`evals/results/<run>/summary.md`).

## Configuration and working rules given to the assistant

- The assistant worked as a senior engineer and mentor, in small, reviewable steps, explaining decisions in plain language.
- No package installs, virtual-environment changes or system software. Missing dependencies were to be reported, and the student installs them.
- `.env` was never to be read aloud, printed, copied or committed. Tests and the notebook's first part build settings with `_env_file=None`.
- No commits, pushes or merges without an explicit request. Work stayed on feature branches; `main` stays stable.
- No fabricated corpus details, test results or evaluation numbers. Every number in the docs comes from a run that was actually executed.
- Decisions that affect cost, privacy, architecture or keys were asked as multiple-choice questions, with a recommended option.
- Live API calls (Gemini, Groq) only with explicit approval. They were approved once, for the evaluation run.

## Prompts

The main prompts, in order (shortened where marked; the decisions themselves are in DESIGN.md):

1. **Planning.** "Plan my Production RAG Service portfolio project", with the assignment brief attached. It asked for a senior architect/mentor role, planning only (no code, edits or installs), FastAPI with plain HTML/CSS, focused questions in small groups with open items marked, and a 13-section plan.
2. **Question format.** Ask planning questions through the multiple-choice popup in small groups with a recommended option and "Other". Don't guess important decisions.
3. **Sharing.** "Put it on a shareable doc or page". This produced the plan page.
4. **Implementation start.** "Start implementing my Production RAG Service project…": the role, the security rules above, the full requirements list, reviewable steps, and "Do not claim evaluations or tests passed until they have actually been run."
5. **Ingestion, part 1.** "Continue from the current repository state and follow the approved project plan…". Create `feature/ingestion`, then implement PDF/DOCX validation, extraction and chunking with page and section metadata, configurations A (1000/150) and B (500/75), tests and notebook part 1.
6. **Embeddings and storage** (commit `494ef3b`, made in a separate session; see that session's chat history for the exact prompt). Gemini embeddings, Chroma, the SQLite registry, duplicate detection, deletion and the document endpoints, with a fake embedder in the tests.
7. **The rest of the project.** "Continue implementing my Production RAG Service in the current repository…": retrieval, the ask endpoint with citations and refusals, guardrails, the HTML/CSS frontend, the evaluation workflow, README, tests with fake LLM and embedder clients, and one commit at the end.

Answers given in the multiple-choice questions (all recorded as decisions in DESIGN.md): the corpus; Groq for answers with a different Groq model as judge; Gemini embeddings; Chroma plus SQLite; own code plus the LangChain text splitter only; Jinja2 and Pico.css with no JavaScript; labelled general knowledge; refusal rules; which bonuses to attempt; and approval to run the live evaluation.

## Chat history

Claude Code keeps the full session transcripts locally. They are submitted separately with the project, not committed to the repository, because they contain local file paths.

## What I checked and understood myself

<!-- Fill in before submission: which parts you reviewed line by line, what you changed, and anything the assistant got wrong that you corrected. -->

- [ ] Read every module in `app/` and can explain its role in the flow.
- [ ] Reviewed the golden set questions and their sources against the documents.
- [ ] Hand-labelled the judge calibration sample (`evals/judge_calibration.csv`).
- [ ] Re-ran the tests and the evaluation myself.
