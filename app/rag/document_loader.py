import re
from pathlib import Path
from typing import Any


def load_markdown_docs(docs_dir: str) -> list[dict[str, Any]]:
    """
    加载 medical_docs 目录下的 Markdown 文件，
    按二级标题（##）切分，返回 chunk 列表。
    """
    chunks = []
    docs_path = Path(docs_dir)
    for md_file in sorted(docs_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        file_chunks = _split_by_heading(text, md_file.name, str(md_file))
        chunks.extend(file_chunks)
    return chunks


def _split_by_heading(text: str, filename: str, source: str) -> list[dict[str, Any]]:
    doc_title = _extract_doc_title(text, filename)
    sections = re.split(r"\n(?=##\s)", text)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"^#+\s+(.+)", section)
        section_title = heading_match.group(1).strip() if heading_match else doc_title
        chunk_id = f"{filename.replace('.md', '')}_{i}"
        chunks.append(
            {
                "id": chunk_id,
                "text": section,
                "metadata": {
                    "title": section_title,
                    "doc_title": doc_title,
                    "source": source,
                    "filename": filename,
                    "knowledge_type": _infer_knowledge_type(filename),
                },
            }
        )
    return chunks


def _extract_doc_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback.replace(".md", "")


def _infer_knowledge_type(filename: str) -> str:
    mapping = {
        "fever": "symptom",
        "cough": "symptom",
        "chest_pain": "emergency",
        "abdominal_pain": "symptom",
        "headache": "symptom",
        "rash": "symptom",
        "dyspnea": "emergency",
        "diarrhea": "symptom",
        "joint_pain": "symptom",
        "trauma": "treatment",
        "eye_discomfort": "symptom",
        "sore_throat": "symptom",
        "hypertension_symptoms": "disease",
        "allergy_reaction": "emergency",
        "diabetes_reminders": "medication_safety",
    }
    for key, ktype in mapping.items():
        if key in filename:
            return ktype
    return "general"
