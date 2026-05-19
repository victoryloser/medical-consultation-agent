from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    advice_generate_node,
    emergency_check_node,
    emergency_response_node,
    image_analysis_node,
    input_check_node,
    report_node,
    retrieval_node,
    risk_assess_node,
    safety_review_node,
    symptom_extract_node,
)
from app.agents.state import ConsultationState


def _route_after_input_check(state: ConsultationState) -> str:
    if state.get("final_report"):
        return "report"
    return "image_analysis"


def _route_after_emergency_check(state: ConsultationState) -> str:
    if state.get("emergency_flag"):
        return "emergency_response"
    return "retrieval"


def build_graph():
    graph = StateGraph(ConsultationState)

    graph.add_node("input_check", input_check_node)
    graph.add_node("image_analysis", image_analysis_node)
    graph.add_node("symptom_extract", symptom_extract_node)
    graph.add_node("emergency_check", emergency_check_node)
    graph.add_node("emergency_response", emergency_response_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("risk_assess", risk_assess_node)
    graph.add_node("advice_generate", advice_generate_node)
    graph.add_node("safety_review", safety_review_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("input_check")

    graph.add_conditional_edges(
        "input_check",
        _route_after_input_check,
        {"report": "report", "image_analysis": "image_analysis"},
    )

    graph.add_edge("image_analysis", "symptom_extract")
    graph.add_edge("symptom_extract", "emergency_check")

    graph.add_conditional_edges(
        "emergency_check",
        _route_after_emergency_check,
        {"emergency_response": "emergency_response", "retrieval": "retrieval"},
    )

    graph.add_edge("emergency_response", "safety_review")

    graph.add_edge("retrieval", "risk_assess")
    graph.add_edge("risk_assess", "advice_generate")
    graph.add_edge("advice_generate", "safety_review")

    graph.add_edge("safety_review", "report")
    graph.add_edge("report", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_consultation_graph():
    return build_graph()
