"""Domain modules of the Immigration Document Processor.

Layering (top imports bottom, never the other way round)::

    app.py                 Streamlit presentation layer
    modules/pipeline.py    orchestration / use cases
    modules/{ocr,classifier,pdf_splitter,renamer,folder_manager,export,validation}
    config.py              settings + logging
"""

__all__ = [
    "classifier",
    "export",
    "folder_manager",
    "ocr",
    "pdf_splitter",
    "pipeline",
    "renamer",
    "validation",
]
