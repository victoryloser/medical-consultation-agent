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
