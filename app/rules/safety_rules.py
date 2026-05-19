import re
from functools import lru_cache

from app.config import get_config

DANGEROUS_PATTERNS = [
    (r"确诊为.{0,20}", "不应给出确诊结论"),
    (r"你得了.{0,20}", "不应替代医生诊断"),
    (r"一定是.{0,20}病", "不应给出确定性诊断"),
    (r"\d+\s*(mg|毫克|片|粒).{0,10}(每天|每日|每次)", "不应给出处方药具体剂量"),
    (r"不用去医院", "不应建议延误就医"),
    (r"不需要就医", "不应建议延误就医"),
]

DISCLAIMER = "\n\n---\n> **免责声明：** 本建议仅供健康咨询参考，不能替代医生诊断。"


class SafetyFilter:

    def __init__(self, extra_keywords: list[str] | None = None):
        self.extra_keywords = extra_keywords or []

    def review(self, advice: str) -> tuple[str, list[str]]:
        """
        返回 (safe_advice, warnings)
        对危险内容进行标注提醒，不直接删除（保留上下文可读性）。
        """
        warnings = []
        safe_advice = advice

        for pattern, reason in DANGEROUS_PATTERNS:
            if re.search(pattern, advice):
                warnings.append(f"内容包含「{reason}」，已提示用户注意")

        for kw in self.extra_keywords:
            if kw in advice:
                warnings.append(f"内容包含敏感词「{kw}」，请以医生建议为准")

        if not safe_advice.strip().endswith(DISCLAIMER.strip()):
            safe_advice = safe_advice + DISCLAIMER

        return safe_advice, warnings


@lru_cache(maxsize=1)
def get_safety_filter() -> SafetyFilter:
    cfg = get_config()
    return SafetyFilter(extra_keywords=cfg.rules.safety_keywords)
