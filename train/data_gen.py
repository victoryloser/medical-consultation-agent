"""
训练数据生成脚本
将 data/test_cases.jsonl 的 30 条种子样本扩充到 300+ 条。

策略：
  1. 模板变体  —— 对每条种子按 6 种表达模板改写，不调用 LLM，快速生成
  2. LLM 增强  —— 调用项目已有客户端，对每类风险等级追加新案例（可选）

输出：
  data/train_sft.jsonl      —— SFT 格式（messages 列表）
  data/train_reward.jsonl   —— GRPO/DPO 格式（prompt + label）

用法：
    python train/data_gen.py
    python train/data_gen.py --llm-augment    # 同时调用 LLM 生成额外样本
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED_PATH   = Path("data/test_cases.jsonl")
SFT_OUT     = Path("data/train_sft.jsonl")
REWARD_OUT  = Path("data/train_reward.jsonl")

SYSTEM_PROMPT = (
    "你是专业医疗风险评估助手。根据用户描述的症状，输出 JSON 格式的评估结果，"
    "字段：risk_level（low/medium/high/emergency）、emergency_flag（true/false）、"
    "reason（一句话简要说明）。只输出 JSON，不要其他内容。"
)

# 6 种口语化改写模板，{s} 为原始症状描述
TEMPLATES = [
    "{s}",
    "我最近{s}，不知道严不严重",
    "请问{s}需要去医院吗",
    "{s}，已经有一段时间了",
    "家人{s}，应该怎么处理",
    "{s}，很担心是不是大问题",
]


def load_seeds() -> list[dict]:
    with open(SEED_PATH, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def make_assistant_output(case: dict) -> str:
    reason_map = {
        "low":       "症状轻微，可先观察，必要时门诊就诊",
        "medium":    "症状需关注，建议近期就医评估",
        "high":      "症状较严重，建议尽快就医",
        "emergency": "存在急症风险，需立即急诊或拨打 120",
        "unknown":   "输入无效，无法评估",
    }
    return json.dumps({
        "risk_level":     case["expected_risk"],
        "emergency_flag": case["expected_emergency"],
        "reason":         reason_map.get(case["expected_risk"], ""),
    }, ensure_ascii=False)


def template_augment(seeds: list[dict]) -> list[dict]:
    samples = []
    for case in seeds:
        text = case["input"].strip()
        if not text:
            continue
        for tpl in TEMPLATES:
            new_text = tpl.replace("{s}", text)
            samples.append({
                "input":              new_text,
                "expected_risk":      case["expected_risk"],
                "expected_emergency": case["expected_emergency"],
            })
    return samples


def llm_augment(seeds: list[dict], n_per_level: int = 15) -> list[dict]:
    """调用项目 LLM 客户端为每个风险等级生成新案例。"""
    from app.clients import get_text_client
    from app.config import get_config
    cfg    = get_config()
    client = get_text_client(cfg)

    level_examples: dict[str, list[str]] = {
        "low": [], "medium": [], "high": [], "emergency": []
    }
    for s in seeds:
        lvl = s["expected_risk"]
        if lvl in level_examples:
            level_examples[lvl].append(s["input"])

    new_samples = []
    for level, examples in level_examples.items():
        if not examples:
            continue
        eg_str = "\n".join(f"- {e}" for e in examples[:5])
        prompt = (
            f"以下是风险等级为「{level}」的医疗问询示例：\n{eg_str}\n\n"
            f"请仿照上述风格，生成 {n_per_level} 条不同的新问询（每行一条，只输出问询文本）："
        )
        try:
            raw = client.chat([{"role": "user", "content": prompt}], temperature=0.9)
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("-·• 0123456789.")
                if len(line) > 4:
                    new_samples.append({
                        "input":              line,
                        "expected_risk":      level,
                        "expected_emergency": level == "emergency",
                    })
        except Exception as e:
            print(f"  LLM 生成 [{level}] 失败: {e}")

    return new_samples


def to_sft_format(sample: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": sample["input"]},
            {"role": "assistant", "content": make_assistant_output(sample)},
        ]
    }


def to_reward_format(sample: dict) -> dict:
    return {
        "prompt":             sample["input"],
        "expected_risk":      sample["expected_risk"],
        "expected_emergency": sample["expected_emergency"],
    }


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {len(records)} 条 → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-augment", action="store_true",
                        help="同时调用 LLM 生成额外样本（需服务在运行）")
    parser.add_argument("--n-per-level", type=int, default=15)
    args = parser.parse_args()

    seeds   = load_seeds()
    samples = template_augment(seeds)
    print(f"模板扩充: {len(seeds)} → {len(samples)} 条")

    if args.llm_augment:
        extra = llm_augment(seeds, n_per_level=args.n_per_level)
        samples += extra
        print(f"LLM 增强: +{len(extra)} 条，合计 {len(samples)} 条")

    random.seed(42)
    random.shuffle(samples)

    write_jsonl(SFT_OUT,    [to_sft_format(s)    for s in samples])
    write_jsonl(REWARD_OUT, [to_reward_format(s) for s in samples])
    print(f"\n完成。SFT 样本: {len(samples)} 条")


if __name__ == "__main__":
    main()
