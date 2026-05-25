# Windows GBK 环境下 trl 读取内置 Jinja 模板时默认编码为 gbk，需在任何 trl 导入前 patch
import pathlib
_orig_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors or "strict")
pathlib.Path.read_text = _utf8_read_text

# PyTorch < 2.6.0 未将 FSDPModule 挂到 torch.distributed.fsdp，TRL 1.4.0 需要它
try:
    from torch.distributed.fsdp import FSDPModule  # noqa: F401
except ImportError:
    import torch.distributed.fsdp as _fsdp
    try:
        from torch.distributed._composable.fsdp import FSDPModule
        _fsdp.FSDPModule = FSDPModule
    except ImportError:
        _fsdp.FSDPModule = object  # dummy，不使用 FSDP 时无影响

"""
GRPO 训练脚本
用「可验证规则奖励」强化急症识别 + 风险分级，无需单独 Reward Model。

奖励设计（每条 response 独立打分，组内相对排序）：
  risk_level 完全正确      +1.0
  risk_level 相差 ±1 级    +0.3
  risk_level 错误           0.0
  emergency 正确            +1.0
  emergency 漏报（FN）      -2.0   ← 医疗场景下漏报惩罚加重
  emergency 误报（FP）      -0.5
  输出格式合法（valid JSON） +0.2   ← 鼓励格式遵从

前置步骤：
    1. python train/data_gen.py
    2. python train/sft.py           （建议先跑 SFT 再做 GRPO）
    3. pip install -r train/requirements.txt

用法：
    python train/grpo.py
    python train/grpo.py --model output/sft --output output/grpo
"""
import argparse
import json
import re
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

REWARD_DATA = Path("data/train_reward.jsonl")

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3, "unknown": -1}

SYSTEM_PROMPT = (
    "你是专业医疗风险评估助手。根据用户描述的症状，输出 JSON 格式的评估结果，"
    "字段：risk_level（low/medium/high/emergency）、emergency_flag（true/false）、"
    "reason（一句话简要说明）。只输出 JSON，不要其他内容。"
)

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)


# ── 奖励函数 ──────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict | None:
    """从模型输出中提取 JSON，兼容 markdown code block。"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def compute_rewards(prompts: list[str],
                    completions: list[str],
                    expected_risks: list[str],
                    expected_emergencies: list[bool]) -> list[float]:
    rewards = []
    for completion, exp_risk, exp_emg in zip(
        completions, expected_risks, expected_emergencies
    ):
        parsed = _parse_response(completion)
        if parsed is None:
            rewards.append(-1.0)
            continue

        reward = 0.2  # 格式合法奖励

        # 风险等级得分
        actual_risk = parsed.get("risk_level", "unknown")
        exp_idx = RISK_ORDER.get(exp_risk, -1)
        act_idx = RISK_ORDER.get(actual_risk, -1)
        diff = abs(exp_idx - act_idx) if exp_idx != -1 and act_idx != -1 else 999
        if diff == 0:
            reward += 1.0
        elif diff == 1:
            reward += 0.3

        # 急症识别得分
        actual_emg = bool(parsed.get("emergency_flag", False))
        if actual_emg == exp_emg:
            reward += 1.0
            if exp_emg:          # 正确识别急症额外加分
                reward += 0.5
        elif exp_emg and not actual_emg:
            reward -= 2.0        # 漏报重罚
        else:
            reward -= 0.5        # 误报轻罚

        rewards.append(round(reward, 3))

    return rewards


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_dataset(path: Path, tokenizer) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": r["prompt"]},
            ]
            records.append({
                "prompt":               tokenizer.apply_chat_template(
                                            messages, tokenize=False,
                                            add_generation_prompt=True),
                "expected_risk":        r["expected_risk"],
                "expected_emergency":   r["expected_emergency"],
            })
    return Dataset.from_list(records)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="output/sft",
                        help="SFT 微调后的模型路径，或原始 Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data",       default=str(REWARD_DATA))
    parser.add_argument("--output",     default="output/grpo")
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--batch",      type=int,   default=2)
    parser.add_argument("--group-size", type=int,   default=4,
                        help="每条 prompt 采样的 response 数量（GRPO 组大小）")
    args = parser.parse_args()

    print(f"[GRPO] 基础模型: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BNBCONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    dataset = load_dataset(Path(args.data), tokenizer)

    # GRPOTrainer 要求 reward_funcs 接收 (prompts, completions, **kwargs)
    def reward_fn(prompts, completions, expected_risk, expected_emergency, **_):
        return compute_rewards(prompts, completions, expected_risk, expected_emergency)

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_generations=args.group_size,
        max_completion_length=128,
        temperature=0.9,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=LORA_CONFIG,
    )

    print("\n[GRPO] 开始训练...")
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n[GRPO] 权重已保存至 {args.output}")


if __name__ == "__main__":
    main()
