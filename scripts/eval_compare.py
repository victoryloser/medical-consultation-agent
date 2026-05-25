"""
微调前后指标对比评估脚本
  - baseline : 调用已运行的 FastAPI（原始 qwen2.5:7b）
  - fine-tuned: 本地加载合并后的模型直接推理

对比维度：
  风险等级严格准确率 / 宽松准确率(±1)
  急症识别 Precision / Recall / F1
  逐条 diff（只显示两者结果不同的样本）

用法：
    # 先启动后端（baseline），再运行对比
    python scripts/eval_compare.py --ft-model output/merged
    python scripts/eval_compare.py --ft-model output/merged --api-url http://localhost:8000
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEST_CASES_PATH = Path("data/test_cases.jsonl")
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3, "unknown": -1}

SYSTEM_PROMPT = (
    "你是专业医疗风险评估助手。根据用户描述的症状，输出 JSON 格式的评估结果，"
    "字段：risk_level（low/medium/high/emergency）、emergency_flag（true/false）、"
    "reason（一句话简要说明）。只输出 JSON，不要其他内容。"
)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_cases() -> list[dict]:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        content = f.read().strip()
    return json.loads(content) if content.startswith("[") else \
           [json.loads(l) for l in content.splitlines() if l.strip()]


# ── Baseline：调用 FastAPI ────────────────────────────────────────────────────

def call_api(base_url: str, text: str) -> dict:
    if not text.strip():
        return {"_skip": True}
    try:
        r = requests.post(
            f"{base_url}/api/consultation",
            data={"text": text, "model_provider": "auto"},
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
        return {"risk_level": body.get("risk_level", "unknown"),
                "emergency_flag": body.get("emergency_flag", False)}
    except Exception as e:
        return {"_error": str(e)}


# ── Fine-tuned：本地推理 ──────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*?\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


class LocalModel:
    def __init__(self, model_path: str):
        print(f"[FT] 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("[FT] 模型加载完成\n")

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"_skip": True}
        print("    [FT] 构建 prompt...", flush=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("    [FT] tokenize...", flush=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        print(f"    [FT] input_ids shape={inputs['input_ids'].shape} device={inputs['input_ids'].device}", flush=True)
        print("    [FT] generate...", flush=True)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        print("    [FT] decode...", flush=True)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"    [FT] 原始输出: {raw[:80]}", flush=True)
        parsed = _parse_json(raw)
        if parsed is None:
            return {"_error": f"JSON解析失败: {raw[:80]}"}
        return {
            "risk_level":    parsed.get("risk_level", "unknown"),
            "emergency_flag": bool(parsed.get("emergency_flag", False)),
        }


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], label: str) -> dict:
    valid = [r for r in results if not r.get("_skip") and not r.get("_error")]
    n = len(valid)
    if n == 0:
        return {}

    risk_correct  = sum(1 for r in valid if r["pred_risk"] == r["exp_risk"])
    loose_correct = sum(
        1 for r in valid
        if abs(RISK_ORDER.get(r["pred_risk"], -1) -
               RISK_ORDER.get(r["exp_risk"],  -1)) <= 1
    )
    tp = sum(1 for r in valid if r["pred_emg"] and r["exp_emg"])
    fp = sum(1 for r in valid if r["pred_emg"] and not r["exp_emg"])
    fn = sum(1 for r in valid if not r["pred_emg"] and r["exp_emg"])

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if precision + recall > 0 else 0.0

    return {
        "label":        label,
        "n":            n,
        "risk_strict":  risk_correct / n,
        "risk_loose":   loose_correct / n,
        "emg_precision": precision,
        "emg_recall":   recall,
        "emg_f1":       f1,
    }


def print_comparison(base_m: dict, ft_m: dict):
    def delta(a, b):
        d = b - a
        sign = "+" if d >= 0 else ""
        color = "\033[92m" if d > 0.005 else ("\033[91m" if d < -0.005 else "")
        reset = "\033[0m" if color else ""
        return f"{color}{sign}{d:.1%}{reset}"

    print(f"\n{'='*62}")
    print(f"  {'指标':<22} {'Baseline':>10} {'Fine-tuned':>10} {'变化':>8}")
    print(f"{'='*62}")
    rows = [
        ("风险严格准确率",  "risk_strict"),
        ("风险宽松准确率±1","risk_loose"),
        ("急症 Precision",  "emg_precision"),
        ("急症 Recall",     "emg_recall"),
        ("急症 F1",         "emg_f1"),
    ]
    for name, key in rows:
        b, f = base_m[key], ft_m[key]
        print(f"  {name:<22} {b:>10.1%} {f:>10.1%} {delta(b,f):>8}")
    print(f"{'='*62}\n")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft-model",  default="output/merged")
    parser.add_argument("--api-url",   default="http://localhost:8000")
    parser.add_argument("--api-delay", type=float, default=0.3)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过 API 调用，直接加载固化的 baseline 结果")
    args = parser.parse_args()

    cases = load_cases()

    print(f"{'='*62}", flush=True)
    print(f"  评估开始，共 {len(cases)} 条用例", flush=True)
    print(f"{'='*62}\n", flush=True)

    # ── Phase 1: Baseline（先收集所有 API 结果，此时不占 GPU）──────────────────
    base_preds: list[dict] = []
    if args.skip_baseline:
        print("[Phase 1] 跳过 baseline API 调用（--skip-baseline）\n", flush=True)
        base_preds = [{}] * len(cases)
    else:
        print("[Phase 1] 收集 baseline 结果（调用 FastAPI）...", flush=True)
        for i, case in enumerate(cases, 1):
            text = case.get("input", "")
            print(f"  [{i:02d}] 调用中: {text[:25] or '(空)'}...", flush=True)
            pred = call_api(args.api_url, text)
            base_preds.append(pred)
            status = pred.get("risk_level", pred.get("_error", "skip"))
            print(f"  [{i:02d}] 完成  : → {status}", flush=True)
            if args.api_delay:
                time.sleep(args.api_delay)
        print("[Phase 1] 完成\n", flush=True)

    # ── Phase 2: Fine-tuned（API 全部收完后再加载模型，独占 GPU）──────────────
    print("[Phase 2] 加载微调模型...", flush=True)
    ft_model = LocalModel(args.ft_model)

    ft_preds: list[dict] = []
    print("[Phase 2] 开始推理...", flush=True)
    for i, case in enumerate(cases, 1):
        text = case.get("input", "")
        print(f"  [{i:02d}] {text[:25] or '(空)'}...", flush=True)
        t0 = time.time()
        try:
            pred = ft_model.predict(text)
        except Exception as e:
            print(f"         推理异常: {e}", flush=True)
            pred = {"_error": str(e)}
        ft_ms = int((time.time() - t0) * 1000)
        status = pred.get("risk_level", pred.get("_error", "skip"))
        print(f"         → {status} ({ft_ms}ms)", flush=True)
        ft_preds.append(pred)
    print(f"[Phase 2] 完成\n", flush=True)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    base_results, ft_results = [], []
    for case, base_pred, ft_pred in zip(cases, base_preds, ft_preds):
        exp_risk = case.get("expected_risk", "unknown")
        exp_emg  = case.get("expected_emergency", False)
        base_r = {"exp_risk": exp_risk, "exp_emg": exp_emg,
                  "pred_risk": base_pred.get("risk_level", "unknown"),
                  "pred_emg":  base_pred.get("emergency_flag", False),
                  **{k: v for k, v in base_pred.items() if k.startswith("_")}}
        ft_r   = {"exp_risk": exp_risk, "exp_emg": exp_emg,
                  "pred_risk": ft_pred.get("risk_level", "unknown"),
                  "pred_emg":  ft_pred.get("emergency_flag", False),
                  **{k: v for k, v in ft_pred.items() if k.startswith("_")}}
        base_results.append(base_r)
        ft_results.append(ft_r)

        if not args.skip_baseline:
            b_ok = "✓" if base_r["pred_risk"] == exp_risk else "✗"
            f_ok = "✓" if ft_r["pred_risk"]   == exp_risk else "✗"
            diff = " ← diff" if b_ok != f_ok else ""
            print(f"  base {b_ok} {base_r['pred_risk']:<9} "
                  f"ft {f_ok} {ft_r['pred_risk']:<9}{diff}", flush=True)

    ft_m = compute_metrics(ft_results, "Fine-tuned")
    if args.skip_baseline:
        print(f"\n{'='*62}")
        print(f"  Fine-tuned 结果（n={ft_m.get('n',0)}）")
        print(f"{'='*62}")
        for name, key in [("风险严格准确率","risk_strict"),("风险宽松准确率±1","risk_loose"),
                          ("急症 Precision","emg_precision"),("急症 Recall","emg_recall"),
                          ("急症 F1","emg_f1")]:
            print(f"  {name:<22} {ft_m[key]:>10.1%}")
        print(f"{'='*62}\n")
        out = {"finetuned": ft_m}
    else:
        base_m = compute_metrics(base_results, "Baseline")
        print_comparison(base_m, ft_m)
        out = {"baseline": base_m, "finetuned": ft_m}

    Path("eval").mkdir(exist_ok=True)
    Path("eval/compare_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("详细结果已保存至 eval/compare_result.json")


if __name__ == "__main__":
    main()
