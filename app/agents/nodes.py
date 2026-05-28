import json
import re
from typing import Any

from app.agents.prompts import (
    ADVICE_GENERATE_PROMPT,
    CONSENSUS_PROMPT,
    EMERGENCY_CONFIRM_PROMPT,
    FACT_CHECK_PROMPT,
    RISK_ASSESS_PROMPT,
    SYMPTOM_EXTRACT_PROMPT,
)
from app.agents.state import ConsultationState
from app.clients import get_text_client, get_vision_client
from app.config import get_config
from app.rag.retriever import get_retriever
from app.rules.emergency_embedding import get_emergency_embedding_retriever
from app.rules.emergency_rules import get_emergency_engine
from app.rules.safety_rules import get_safety_filter


def input_check_node(state: ConsultationState) -> dict:
    text = (state.get("user_text") or "").strip()
    if not text:
        return {
            "errors": ["用户输入为空"],
            "risk_level": "unknown",
            "final_report": "请输入症状描述后再发起问诊。",
        }
    if len(text) > 2000:
        return {"user_text": text[:2000], "errors": ["输入过长，已截断"]}
    return {}


def image_analysis_node(state: ConsultationState) -> dict:
    if not state.get("image_path"):
        return {"image_analysis": None}
    cfg = get_config()
    client = get_vision_client(cfg)
    try:
        result = client.vision_chat(
            text="请简要描述图片中可见的医学相关体征或外观特征，用中文描述，不要做疾病确诊判断，不超过200字。",
            image_path=state["image_path"],
        )
        return {"image_analysis": result}
    except Exception as e:
        return {"image_analysis": None, "errors": [f"图像分析失败: {e}"]}


def symptom_extract_node(state: ConsultationState) -> dict:
    context = state.get("user_text", "")
    if state.get("image_analysis"):
        context += f"\n图像描述：{state['image_analysis']}"
    cfg = get_config()
    client = get_text_client(cfg)
    try:
        prompt = SYMPTOM_EXTRACT_PROMPT.format(context=context)
        raw = client.chat([{"role": "user", "content": prompt}], temperature=0.1)
        symptoms = _parse_symptom_list(raw)
        return {"symptoms": symptoms}
    except Exception as e:
        return {"symptoms": [], "errors": [f"症状抽取失败: {e}"]}


def emergency_check_node(state: ConsultationState) -> dict:
    all_text = (state.get("user_text") or "") + " " + " ".join(state.get("symptoms") or [])

    # 链路一：关键词规则引擎（高精度，O(n) 低延迟）
    engine = get_emergency_engine()
    is_emergency, reason, risk = engine.check(all_text)
    if is_emergency:
        return {"emergency_flag": True, "emergency_reason": reason, "risk_level": risk}

    # 链路二：Embedding 语义相似度
    emb_retriever = get_emergency_embedding_retriever()
    hit, category, emb_risk, score = emb_retriever.check(all_text)

    if not hit:
        return {"emergency_flag": False, "emergency_reason": None}

    cfg = get_config()
    high_threshold = cfg.rules.emergency_embedding_high_threshold

    # 高置信区间：直接判定
    if score >= high_threshold:
        reason = f"语义匹配到高风险场景「{category}」（相似度 {score:.2f}）"
        return {"emergency_flag": True, "emergency_reason": reason, "risk_level": emb_risk}

    # 灰色地带：LLM 二次确认，避免误报
    client = get_text_client(cfg)
    try:
        prompt = EMERGENCY_CONFIRM_PROMPT.format(
            text=state.get("user_text", ""),
            category=category,
        )
        raw = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        confirmed = raw.strip().lower().startswith("yes")
    except Exception:
        confirmed = False

    if confirmed:
        reason = f"语义匹配到高风险场景「{category}」（相似度 {score:.2f}，LLM 确认）"
        return {"emergency_flag": True, "emergency_reason": reason, "risk_level": emb_risk}

    return {"emergency_flag": False, "emergency_reason": None}


def emergency_response_node(state: ConsultationState) -> dict:
    reason = state.get("emergency_reason") or "检测到高风险症状"
    action = f"{reason}\n\n**请立即前往急诊科或拨打急救电话（120）。请勿自行驾车前往医院。**"
    return {
        "advice": action,
        "department_suggestion": ["急诊科"],
        "retrieved_docs": [],
    }


def retrieval_node(state: ConsultationState) -> dict:
    query = (state.get("user_text") or "") + " " + " ".join(state.get("symptoms") or [])
    retriever = get_retriever()
    try:
        docs = retriever.search(query.strip())
        return {"retrieved_docs": docs}
    except Exception as e:
        return {"retrieved_docs": [], "errors": [f"知识检索失败: {e}"]}


def risk_assess_node(state: ConsultationState) -> dict:
    if state.get("risk_level") in ("high", "emergency"):
        return {}
    symptoms = state.get("symptoms") or []
    docs = state.get("retrieved_docs") or []
    knowledge_summary = "\n".join(
        d.get("content", "")[:100] for d in docs[:3]
    )
    cfg = get_config()
    client = get_text_client(cfg)
    try:
        prompt = RISK_ASSESS_PROMPT.format(
            symptoms=", ".join(symptoms) or "无明显症状",
            knowledge_summary=knowledge_summary or "无参考知识",
        )
        raw = client.chat([{"role": "user", "content": prompt}], temperature=0.1)
        level = _parse_risk_level(raw)
        return {"risk_level": level}
    except Exception:
        return {"risk_level": "medium"}


def advice_generate_node(state: ConsultationState) -> dict:
    symptoms = state.get("symptoms") or []
    docs = state.get("retrieved_docs") or []
    docs_text = _format_docs(docs)
    cfg = get_config()
    client = get_text_client(cfg)
    try:
        prompt = ADVICE_GENERATE_PROMPT.format(
            symptoms=", ".join(symptoms) or "无明显症状",
            risk_level=state.get("risk_level", "medium"),
            image_analysis=state.get("image_analysis") or "无图像",
            retrieved_docs=docs_text,
        )
        advice = client.chat([{"role": "user", "content": prompt}], temperature=0.5)
        departments = _extract_departments(advice)
        return {"advice": advice, "department_suggestion": departments}
    except Exception as e:
        return {
            "advice": "建议就医咨询，具体建议生成失败。",
            "department_suggestion": ["普通内科"],
            "errors": [f"建议生成失败: {e}"],
        }


# ── 三层验证节点（参照 WSI-Agents 验证机制）─────────────────────────────────

def fact_check_node(state: ConsultationState) -> dict:
    """Fact Agent：对生成建议做知识库二次核验，输出置信分。"""
    advice = state.get("advice") or ""
    docs = state.get("retrieved_docs") or []
    if not advice or not docs:
        return {"fact_confidence": 1.0}
    knowledge = "\n".join(d.get("content", "")[:150] for d in docs[:4])
    cfg = get_config()
    client = get_text_client(cfg)
    try:
        prompt = FACT_CHECK_PROMPT.format(advice=advice[:500], knowledge=knowledge)
        raw = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        score = float(re.search(r"[01](?:\.\d+)?", raw.strip()).group())
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.8
    return {"fact_confidence": round(score, 3)}


def consensus_node(state: ConsultationState) -> dict:
    """Consensus Agent：调用独立 LLM 对风险等级做第二意见，计算一致性。"""
    if state.get("emergency_flag"):
        return {"consensus_agreement": 1.0}
    symptoms = state.get("symptoms") or []
    user_text = state.get("user_text") or ""
    primary_risk = state.get("risk_level", "medium")
    cfg = get_config()
    client = get_text_client(cfg)
    try:
        prompt = CONSENSUS_PROMPT.format(
            symptoms=", ".join(symptoms) or "无明显症状",
            user_text=user_text[:200],
        )
        raw = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        second_risk = _parse_risk_level(raw)
        agreement = 1.0 if second_risk == primary_risk else 0.5
        # 两次结果不一致时以较高风险为准，保守原则
        if agreement < 1.0:
            risk_order = {"low": 0, "medium": 1, "high": 2, "emergency": 3}
            safer_risk = max(primary_risk, second_risk,
                             key=lambda r: risk_order.get(r, 0))
            return {"consensus_agreement": agreement, "risk_level": safer_risk}
    except Exception:
        agreement = 1.0
    return {"consensus_agreement": agreement}


def consistency_check_node(state: ConsultationState) -> dict:
    """Logic Agent：检查 emergency_flag 与 risk_level 是否逻辑一致，强制对齐。"""
    emergency_flag = state.get("emergency_flag", False)
    risk_level = state.get("risk_level", "medium")
    risk_order = {"low": 0, "medium": 1, "high": 2, "emergency": 3}
    fixed = False

    # 矛盾一：标记了急症但风险等级不是 emergency
    if emergency_flag and risk_order.get(risk_level, 0) < 3:
        risk_level = "emergency"
        fixed = True

    # 矛盾二：风险是 emergency 但没有标记 emergency_flag
    if risk_level == "emergency" and not emergency_flag:
        emergency_flag = True
        fixed = True

    if fixed:
        return {
            "risk_level": risk_level,
            "emergency_flag": emergency_flag,
            "consistency_fixed": True,
            "errors": ["⚠️ Logic Agent：检测到风险等级与急症标志不一致，已自动修正"],
        }
    return {"consistency_fixed": False}


def safety_review_node(state: ConsultationState) -> dict:
    advice = state.get("advice") or ""
    safety_filter = get_safety_filter()
    safe_advice, warnings = safety_filter.review(advice)
    return {"advice": safe_advice, "safety_warnings": warnings}


def report_node(state: ConsultationState) -> dict:
    report = _render_report(state)
    return {"final_report": report}


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_symptom_list(raw: str) -> list[str]:
    raw = raw.strip()
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            return [s.strip() for s in items if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
    items = re.findall(r"[""「」]([^""「」]+)[""「」]|[・•\-]\s*(.+)", raw)
    result = []
    for a, b in items:
        token = (a or b).strip()
        if token:
            result.append(token)
    return result[:10]


def _parse_risk_level(raw: str) -> str:
    raw = raw.strip().lower()
    for level in ("emergency", "high", "medium", "low"):
        if level in raw:
            return level
    return "medium"


def _format_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "暂无相关医学知识。"
    parts = []
    for d in docs[:5]:
        title = d.get("title", "")
        content = d.get("content", "")[:200]
        parts.append(f"**{title}**\n{content}")
    return "\n\n".join(parts)


def _extract_departments(advice: str) -> list[str]:
    candidates = [
        "急诊科", "内科", "外科", "普通内科", "呼吸内科", "心内科", "消化内科",
        "神经内科", "皮肤科", "眼科", "耳鼻喉科", "骨科", "妇科", "儿科",
        "感染科", "肿瘤科", "内分泌科", "风湿科", "泌尿外科", "心理科",
    ]
    found = [d for d in candidates if d in advice]
    return found[:3] if found else ["普通内科"]


def _render_report(state: ConsultationState) -> str:
    risk_map = {
        "low": "低风险 - 可观察，必要时普通门诊就诊",
        "medium": "中风险 - 建议近期就医咨询",
        "high": "高风险 - 建议尽快就医",
        "emergency": "急症风险 - 建议立即急诊或拨打120",
        "unknown": "未知",
    }
    risk_label = risk_map.get(state.get("risk_level", "unknown"), "未知")
    symptoms_text = "、".join(state.get("symptoms") or []) or "暂未识别到明确症状"
    departments_text = "、".join(state.get("department_suggestion") or []) or "普通内科"

    image_section = ""
    if state.get("image_analysis"):
        image_section = f"\n### 2. 图片初步分析\n{state['image_analysis']}\n"

    docs = state.get("retrieved_docs") or []
    refs_text = "\n".join(
        f"- {d.get('title', '')}（{d.get('source', '')}）"
        for d in docs
        if d.get("title")
    ) or "无"

    warnings = state.get("safety_warnings") or []
    warnings_text = "\n".join(f"- {w}" for w in warnings) if warnings else "无特殊提醒"

    # 验证层摘要
    fact_conf = state.get("fact_confidence")
    consensus = state.get("consensus_agreement")
    consistency_fixed = state.get("consistency_fixed", False)
    verification_text = ""
    if fact_conf is not None or consensus is not None:
        lines = []
        if fact_conf is not None:
            lines.append(f"- 知识库一致性（Fact）：{fact_conf:.0%}")
        if consensus is not None:
            lines.append(f"- 双模型共识度（Consensus）：{consensus:.0%}")
        if consistency_fixed:
            lines.append("- 逻辑校验（Logic）：检测到矛盾并已自动修正")
        else:
            lines.append("- 逻辑校验（Logic）：通过")
        verification_text = "\n### 10. 验证层摘要\n" + "\n".join(lines) + "\n"

    emergency_section = ""
    if state.get("emergency_flag"):
        reason = state.get("emergency_reason") or ""
        emergency_section = f"\n> ⚠️ **急症提示：** {reason}\n"

    advice = state.get("advice") or "建议就医咨询。"

    report = f"""## 智能医疗辅助问诊报告

### 1. 用户输入摘要
- 症状描述：{state.get('user_text', '')}
- 是否上传图片：{'是' if state.get('image_path') else '否'}
{image_section}
### 3. 识别到的主要症状
{symptoms_text}

### 4. 风险等级
**{risk_label}**
{emergency_section}
### 5. 建议就医科室
{departments_text}

### 6. 健康咨询建议
{advice}

### 7. 用药安全提醒
{warnings_text}

不建议自行使用处方药，如需用药请咨询医生或药师。

### 8. 参考知识来源
{refs_text}

### 9. 免责声明
本报告仅用于健康咨询和学习演示，**不能替代医生诊断**。如症状严重或持续加重，请及时就医。
{verification_text}"""
    return report.strip()
