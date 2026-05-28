import operator
from typing import Annotated, Any, Optional, TypedDict


class ConsultationState(TypedDict):
    user_text: str
    image_path: Optional[str]
    image_analysis: Optional[str]
    symptoms: Annotated[list[str], operator.add]
    emergency_flag: bool
    emergency_reason: Optional[str]
    retrieved_docs: list[dict[str, Any]]
    risk_level: str                      # low | medium | high | emergency | unknown
    department_suggestion: list[str]
    advice: str
    safety_warnings: Annotated[list[str], operator.add]
    final_report: str
    errors: Annotated[list[str], operator.add]
    # ── 三层验证（参照 WSI-Agents 验证机制）──────────────────────────────────
    fact_confidence: Optional[float]     # Fact Agent：建议与知识库的一致性分 [0,1]
    consensus_agreement: Optional[float] # Consensus Agent：双模型风险等级一致性 [0,1]
    consistency_fixed: Optional[bool]    # Logic Agent：是否修正了逻辑矛盾
