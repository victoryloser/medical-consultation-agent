"""
文档入库脚本：将 data/medical_docs 下的 Markdown 文件向量化并存入 ChromaDB。
运行：python scripts/ingest_docs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_config
from app.rag.document_loader import load_markdown_docs
from app.rag.vector_store import get_vector_store


def main():
    cfg = get_config()
    docs_dir = "data/medical_docs"

    print(f"加载文档目录：{docs_dir}")
    chunks = load_markdown_docs(docs_dir)
    print(f"切分得到 {len(chunks)} 个文本块")

    store = get_vector_store()
    print(f"正在写入 ChromaDB（{cfg.rag.persist_dir}）...")
    store.add_documents(chunks)

    count = store.count()
    print(f"入库完成，当前知识库共 {count} 条记录。")


if __name__ == "__main__":
    main()
