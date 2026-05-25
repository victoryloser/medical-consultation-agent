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
        _fsdp.FSDPModule = object

"""
DPO 训练脚本（含偏好对自动构造）

偏好对构造策略：
  chosen  ── 风险等级正确 + 语言规范（不含确诊/处方）
  rejected── 以下任一：
              a) 风险等级错误（especially 漏报急症）
              b) 包含确诊语言（"确诊为 X"、"你得了 X"）
              c) 给出处方药剂量

分两步运行：
  Step A: 构造偏好对（调用在线服务生成 rejected 样本）
    python train/dpo.py --build-pairs

  Step B: DPO 训练
    python train/dpo.py --train

  或一次完成：
    python train/dpo.py --build-pairs --train
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SFT_DATA   = Path("data/train_sft.jsonl")
PAIRS_OUT  = Path("data/train_dpo.jsonl")

SYSTEM_PROMPT = (
    "你是专业医疗风险评估助手。根据用户描述的症状，输出 JSON 格式的评估结果，"
    "字段：risk_level（low/medium/high/emergency）、emergency_flag（true/false）、"
    "reason（一句话简要说明）。只输出 JSON，不要其他内容。"
)

# safety_rules.py 中已定义的危险模式，在此对齐
REJECT_PATTERNS = [
    re.compile(r"确诊为.{0,20}"),
    re.compile(r"你得了.{0,20}"),
    re.compile(r"一定是.{0,20}病"),
    re.compile(r"\d+\s*(mg|毫克|片|粒).{0,10}(每天|每日|每次)"),
    re.compile(r"不用去医院"),
    re.compile(r"不需要就医"),
]

# 用于生成"故意错误"的 rejected 样本的 prompt 模板
REJECTED_PROMPT_TPL = (
    "以下是用户的症状描述，请故意给出一个错误的风险评估（比如把急症评为 low，"
    "或使用「确诊为 X」这样的语言），输出 JSON：\n{user_input}"
)


def _contains_rejection_pattern(text: str) -> bool:
    return any(p.search(text) for p in REJECT_PATTERNS)


def build_pairs():
    """
    从 train_sft.jsonl 的 chosen 样本出发，
    调用 LLM 生成对应的 rejected 样本，构造 DPO 偏好对。
    """
    from app.clients import get_text_client
    from app.config import get_config
    cfg    = get_config()
    client = get_text_client(cfg)

    pairs = []
    with open(SFT_DATA, encoding="utf-8") as f:
        sft_records = [json.loads(l) for l in f if l.strip()]

    for record in sft_records:
        messages  = record["messages"]
        user_msg  = next(m["content"] for m in messages if m["role"] == "user")
        chosen    = next(m["content"] for m in messages if m["role"] == "assistant")

        # 尝试让 LLM 生成一个刻意错误的输出作为 rejected
        try:
            rejected = client.chat(
                [{"role": "user", "content": REJECTED_PROMPT_TPL.format(
                    user_input=user_msg)}],
                temperature=1.0,
            )
        except Exception:
            # 退回到内置的 rejected 模板（固定错误格式）
            rejected = json.dumps({
                "risk_level":     "low",
                "emergency_flag": False,
                "reason":         f"确诊为普通感冒，不需要就医，每天服用 500mg 布洛芬即可",
            }, ensure_ascii=False)

        pairs.append({
            "prompt":   user_msg,
            "chosen":   chosen,
            "rejected": rejected,
        })

    PAIRS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[DPO] 构造偏好对 {len(pairs)} 条 → {PAIRS_OUT}")


def train_dpo(args):
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )

    # 构建完整 prompt（含 system）供 DPOTrainer 使用
    def format_record(r: dict) -> dict:
        full_prompt = tokenizer.apply_chat_template(
            [{"role": "system",  "content": SYSTEM_PROMPT},
             {"role": "user",    "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": full_prompt,
                "chosen": r["chosen"],
                "rejected": r["rejected"]}

    records = []
    with open(PAIRS_OUT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(format_record(json.loads(line)))
    dataset = Dataset.from_list(records)
    split   = dataset.train_test_split(test_size=0.1, seed=42)

    dpo_config = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=0.1,           # KL 散度系数，越小越偏离参考模型
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora,
    )

    print("\n[DPO] 开始训练...")
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n[DPO] 权重已保存至 {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-pairs", action="store_true")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--model",   default="output/grpo",
                        help="GRPO 微调后的模型路径")
    parser.add_argument("--output",  default="output/dpo")
    parser.add_argument("--epochs",  type=int,   default=2)
    parser.add_argument("--lr",      type=float, default=1e-5)
    parser.add_argument("--batch",   type=int,   default=2)
    args = parser.parse_args()

    if not args.build_pairs and not args.train:
        parser.print_help()
        return

    if args.build_pairs:
        build_pairs()
    if args.train:
        train_dpo(args)


if __name__ == "__main__":
    main()
