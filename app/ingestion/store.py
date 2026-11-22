"""Storage for ingested chunks lives in ``app/storage/``.

* ``app/storage/vector_store.py``: Chroma collection (vectors, text, metadata).
* ``app/storage/registry.py``: SQLite document registry and embedding cache.

``app/services/document_service.py`` calls them in the all-or-nothing order
described in DESIGN.md. This module is kept so the ``ingestion/`` package
mirrors the brief's suggested layout.
"""
