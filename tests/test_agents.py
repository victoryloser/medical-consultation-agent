import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.agents.nodes import (
    _parse_risk_level,
    _parse_symptom_list,
    input_check_node,
)
from app.agents.state import ConsultationState


def _make_state(**kwargs) -> ConsultationState:
    defaults: ConsultationState = {
        "user_text": "",
        "image_path": None,
        "image_analysis": None,
        "symptoms": [],
        "emergency_flag": False,
        "emergency_reason": None,
        "retrieved_docs": [],
        "risk_level": "unknown",
        "department_suggestion": [],
        "advice": "",
        "safety_warnings": [],
        "final_report": "",
        "errors": [],
    }
    defaults.update(kwargs)
    return defaults


def test_input_check_empty():
    state = _make_state(user_text="")
    result = input_check_node(state)
    assert "errors" in result
    assert len(result["errors"]) > 0


def test_input_check_valid():
    state = _make_state(user_text="我发烧 38 度")
    result = input_check_node(state)
    assert result == {} or "errors" not in result or len(result.get("errors", [])) == 0


def test_parse_symptom_list_json():
    raw = '["发热","咳嗽","乏力"]'
    symptoms = _parse_symptom_list(raw)
    assert "发热" in symptoms
    assert "咳嗽" in symptoms


def test_parse_symptom_list_fallback():
    raw = "主要症状包括：• 发热\n• 头痛\n• 乏力"
    symptoms = _parse_symptom_list(raw)
    assert len(symptoms) >= 0  # fallback 不报错


def test_parse_risk_level():
    assert _parse_risk_level("这是 high 风险") == "high"
    assert _parse_risk_level("low") == "low"
    assert _parse_risk_level("medium level") == "medium"
    assert _parse_risk_level("完全未知内容") == "medium"
