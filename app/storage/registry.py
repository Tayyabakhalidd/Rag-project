"""SQLite document registry and successful-embedding cache."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL, file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, page_count INTEGER,
                    section_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
                    chunk_config TEXT NOT NULL, created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('processing','ready'))
                );
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY, vector_json TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def reserve(self, document: dict[str, Any]) -> bool:
        """Atomically claim a hash before doing expensive embedding work."""
        with self._connect() as db:
            try:
                db.execute(
                    """INSERT INTO documents
                    (document_id,sha256,filename,file_type,size_bytes,page_count,section_count,
                    chunk_count,chunk_config,created_at,status)
                    VALUES (:document_id,:sha256,:filename,:file_type,:size_bytes,:page_count,
                    :section_count,:chunk_count,:chunk_config,:created_at,'processing')""",
                    document,
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def mark_ready(self, document_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE documents SET status='ready' WHERE document_id=?", (document_id,))

    def remove(self, document_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
            return cursor.rowcount > 0

    def get(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE document_id=? AND status='ready'", (document_id,)).fetchone()
        return dict(row) if row else None

    def list_ready(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM documents WHERE status='ready' ORDER BY created_at,document_id").fetchall()
        return [dict(row) for row in rows]

    def get_embedding(self, cache_key: str) -> list[float] | None:
        with self._connect() as db:
            row = db.execute("SELECT vector_json FROM embedding_cache WHERE cache_key=?", (cache_key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_embedding(self, cache_key: str, vector: list[float]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO embedding_cache(cache_key,vector_json) VALUES (?,?)",
                (cache_key, json.dumps(vector, separators=(",", ":"))),
            )


def document_record(prepared) -> dict[str, Any]:
    return {
        "document_id": prepared.document_id,
        "sha256": prepared.sha256,
        "filename": prepared.filename,
        "file_type": prepared.file_type,
        "size_bytes": prepared.size_bytes,
        "page_count": prepared.page_count,
        "section_count": prepared.section_count,
        "chunk_count": len(prepared.chunks),
        "chunk_config": prepared.chunk_config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
