import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.rules.emergency_rules import EmergencyRuleEngine


@pytest.fixture
def engine():
    return EmergencyRuleEngine("data/emergency_rules.json")


def test_chest_pain_emergency(engine):
    text = "我胸口突然很痛，还冒冷汗，左肩也疼"
    is_emg, action, risk = engine.check(text)
    assert is_emg is True
    assert risk == "emergency"


def test_normal_text_no_emergency(engine):
    text = "我有点头疼，可能是睡眠不好"
    is_emg, action, risk = engine.check(text)
    assert is_emg is False


def test_breathing_difficulty(engine):
    text = "我呼吸困难，喘不上气来"
    is_emg, action, risk = engine.check(text)
    assert is_emg is True
    assert risk == "emergency"


def test_mental_crisis(engine):
    text = "我不想活了，想死"
    is_emg, action, risk = engine.check(text)
    assert is_emg is True
    assert "120" in action or "援助" in action


def test_high_fever_child(engine):
    text = "宝宝高烧不退，已经两天了"
    is_emg, action, risk = engine.check(text)
    assert is_emg is True
    assert risk in ("high", "emergency")


def test_mild_fever_no_trigger(engine):
    text = "我发烧 37.5 度，有点乏力"
    is_emg, _, _ = engine.check(text)
    assert is_emg is False
