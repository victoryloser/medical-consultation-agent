import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.rag.document_loader import load_markdown_docs


def test_load_docs_count():
    chunks = load_markdown_docs("data/medical_docs")
    assert len(chunks) > 0, "应至少有一个文本块"


def test_load_docs_structure():
    chunks = load_markdown_docs("data/medical_docs")
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk
        assert "title" in chunk["metadata"]
        assert "source" in chunk["metadata"]


def test_doc_coverage():
    chunks = load_markdown_docs("data/medical_docs")
    sources = {c["metadata"]["filename"] for c in chunks}
    required = {"fever.md", "cough.md", "chest_pain.md"}
    assert required.issubset(sources), f"缺少必要文档: {required - sources}"
