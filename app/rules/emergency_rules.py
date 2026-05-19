import json
from functools import lru_cache
from pathlib import Path

from app.config import get_config


class EmergencyRuleEngine:

    def __init__(self, rules_path: str):
        path = Path(rules_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
        else:
            self.rules = []

    def check(self, text: str) -> tuple[bool, str, str]:
        """
        返回 (is_emergency_or_high, action_message, risk_level)
        遍历规则，优先返回 emergency 级别。
        """
        best_risk = None
        best_action = ""

        for rule in self.rules:
            keywords = rule.get("keywords", [])
            match_count = rule.get("match_count", 1)
            hit = sum(1 for kw in keywords if kw in text)
            if hit >= match_count:
                risk = rule.get("risk_level", "high")
                action = rule.get("action", "建议就医。")
                if risk == "emergency":
                    return True, action, "emergency"
                if best_risk != "emergency":
                    best_risk = risk
                    best_action = action

        if best_risk == "high":
            return True, best_action, "high"
        return False, "", ""


@lru_cache(maxsize=1)
def get_emergency_engine() -> EmergencyRuleEngine:
    cfg = get_config()
    return EmergencyRuleEngine(cfg.rules.emergency_rules_path)
