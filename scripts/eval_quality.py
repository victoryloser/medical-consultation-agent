"""
维度二：AI 回答质量评估
对 data/test_cases.jsonl 中的 30 条标注样本发起真实 HTTP 请求，
统计风险等级准确率、急症识别 Precision/Recall/F1，并打印错误样本。

用法：
    python scripts/eval_quality.py
    python scripts/eval_quality.py --url http://localhost:8000 --delay 1.0
"""
import argparse
import json
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
ENDPOINT = "/api/consultation"
TEST_CASES_PATH = Path("data/test_cases.jsonl")

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3, "unknown": -1}


def load_cases(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def call_api(base_url: str, text: str) -> dict | None:
    try:
        resp = requests.post(
            f"{base_url}{ENDPOINT}",
            data={"text": text, "model_provider": "auto"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"_error": str(e)}


def evaluate(cases: list[dict], base_url: str, delay: float) -> dict:
    results = []
    print(f"\n{'='*60}")
    print(f"  评估开始，共 {len(cases)} 条用例，接口: {base_url}{ENDPOINT}")
    print(f"{'='*60}\n")

    for i, case in enumerate(cases, 1):
        text = case.get("input", "")
        expected_risk = case.get("expected_risk", "unknown")
        expected_emg = case.get("expected_emergency", False)

        print(f"[{i:02d}/{len(cases)}] 输入: {text[:40] or '(空)'}...")
        resp = call_api(base_url, text)

        if "_error" in resp:
            print(f"       ✗ 请求失败: {resp['_error']}\n")
            results.append({
                "case": case, "error": resp["_error"],
                "risk_correct": False, "emg_correct": False,
            })
            if delay:
                time.sleep(delay)
            continue

        actual_risk = resp.get("risk_level", "unknown")
        actual_emg = resp.get("emergency_flag", False)
        risk_correct = actual_risk == expected_risk
        emg_correct = actual_emg == expected_emg

        status_risk = "✓" if risk_correct else "✗"
        status_emg = "✓" if emg_correct else "✗"
        print(
            f"       风险 {status_risk} 预期={expected_risk:<9} 实际={actual_risk:<9} | "
            f"急症 {status_emg} 预期={str(expected_emg):<5} 实际={str(actual_emg)}"
        )
        print()

        results.append({
            "case": case,
            "actual_risk": actual_risk,
            "actual_emg": actual_emg,
            "risk_correct": risk_correct,
            "emg_correct": emg_correct,
            "symptoms": resp.get("symptoms", []),
            "department": resp.get("department_suggestion", []),
        })

        if delay:
            time.sleep(delay)

    return _summarize(results)


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    errors = [r for r in results if "error" in r]
    valid = [r for r in results if "error" not in r]

    risk_correct = sum(1 for r in valid if r["risk_correct"])
    emg_correct = sum(1 for r in valid if r["emg_correct"])

    # 急症 Precision / Recall / F1
    tp = sum(1 for r in valid if r["actual_emg"] and r["case"]["expected_emergency"])
    fp = sum(1 for r in valid if r["actual_emg"] and not r["case"]["expected_emergency"])
    fn = sum(1 for r in valid if not r["actual_emg"] and r["case"]["expected_emergency"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 风险等级偏差（允许 ±1 级视为"宽松准确"）
    loose_correct = 0
    for r in valid:
        exp = RISK_ORDER.get(r["case"]["expected_risk"], -1)
        act = RISK_ORDER.get(r.get("actual_risk", "unknown"), -1)
        if abs(exp - act) <= 1:
            loose_correct += 1

    summary = {
        "total": total,
        "api_errors": len(errors),
        "valid": len(valid),
        "risk_accuracy": risk_correct / len(valid) if valid else 0,
        "risk_loose_accuracy": loose_correct / len(valid) if valid else 0,
        "emg_accuracy": emg_correct / len(valid) if valid else 0,
        "emg_precision": precision,
        "emg_recall": recall,
        "emg_f1": f1,
        "wrong_risk": [r for r in valid if not r["risk_correct"]],
        "wrong_emg": [r for r in valid if not r["emg_correct"]],
    }

    _print_summary(summary)
    return summary


def _print_summary(s: dict):
    print(f"\n{'='*60}")
    print("  评估结果汇总")
    print(f"{'='*60}")
    print(f"  总用例数      : {s['total']}")
    print(f"  API 错误数    : {s['api_errors']}")
    print(f"  有效样本数    : {s['valid']}")
    print()
    print(f"  ── 风险等级 ──────────────────────────────")
    print(f"  严格准确率    : {s['risk_accuracy']:.1%}  ({int(s['risk_accuracy']*s['valid'])}/{s['valid']})")
    print(f"  宽松准确率±1  : {s['risk_loose_accuracy']:.1%}  ({int(s['risk_loose_accuracy']*s['valid'])}/{s['valid']})")
    print()
    print(f"  ── 急症识别 ──────────────────────────────")
    print(f"  准确率        : {s['emg_accuracy']:.1%}")
    print(f"  Precision     : {s['emg_precision']:.1%}")
    print(f"  Recall        : {s['emg_recall']:.1%}")
    print(f"  F1            : {s['emg_f1']:.1%}")

    if s["wrong_risk"]:
        print(f"\n  ── 风险等级错误样本 ({len(s['wrong_risk'])} 条) ────────────")
        for r in s["wrong_risk"]:
            inp = r["case"]["input"][:45]
            print(f"  [{r['case']['expected_risk']} → {r.get('actual_risk','?')}] {inp}")

    if s["wrong_emg"]:
        print(f"\n  ── 急症识别错误样本 ({len(s['wrong_emg'])} 条) ────────────")
        for r in s["wrong_emg"]:
            tag = "漏报(FN)" if r["case"]["expected_emergency"] else "误报(FP)"
            inp = r["case"]["input"][:45]
            print(f"  [{tag}] {inp}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="每次请求后的间隔秒数（避免过载本地 Ollama）")
    args = parser.parse_args()

    cases = load_cases(TEST_CASES_PATH)
    evaluate(cases, args.url, args.delay)
