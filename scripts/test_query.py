"""
快速测试检索脚本：输入查询，返回相关知识片段。
运行：python scripts/test_query.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.retriever import get_retriever


def main():
    retriever = get_retriever()
    if not retriever.is_ready():
        print("知识库为空，请先运行 scripts/ingest_docs.py")
        return

    queries = [
        "发烧咳嗽乏力",
        "胸口很痛大汗",
        "皮肤起红点痒",
        "血压高头痛",
    ]

    for query in queries:
        print(f"\n查询：{query}")
        results = retriever.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['title']} | 相似度: {r['score']:.3f} | 来源: {r['source']}")


if __name__ == "__main__":
    main()
