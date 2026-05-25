# Windows GBK 环境下 trl 读取内置 Jinja 模板时默认编码为 gbk，需在任何 trl 导入前 patch
import pathlib
_orig_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors or "strict")
pathlib.Path.read_text = _utf8_read_text

"""
SFT 监督微调脚本
目标任务：症状描述 → 结构化风险评估（risk_level + emergency_flag + reason）
模型：Qwen/Qwen2.5-7B-Instruct（与项目 Ollama 同款权重）
方法：QLoRA（4-bit 量化 + LoRA r=16），单卡 16GB 可跑

前置步骤：
    1. python train/data_gen.py [--llm-augment]
    2. pip install -r train/requirements.txt

用法：
    python train/sft.py
    python train/sft.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3 --output output/sft
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

SFT_DATA = Path("data/train_sft.jsonl")

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # 覆盖 Qwen2.5 的 attention + FFN 投影层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)


def load_dataset(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_messages(tokenizer, example: dict) -> dict:
    """将 messages 列表转换为模型输入的 text 字段。"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data",    default=str(SFT_DATA))
    parser.add_argument("--output",  default="output/sft")
    parser.add_argument("--epochs",  type=int,   default=3)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--max-len", type=int,   default=512)
    args = parser.parse_args()

    print(f"[SFT] 模型: {args.model}")
    print(f"[SFT] 数据: {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BNBCONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    dataset = load_dataset(Path(args.data))
    dataset = dataset.map(
        lambda ex: format_messages(tokenizer, ex),
        remove_columns=["messages"],
    )

    split = dataset.train_test_split(test_size=0.1, seed=42)

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    print("\n[SFT] 开始训练...")
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n[SFT] LoRA 权重已保存至 {args.output}")
    print("后续：用 llama.cpp 或 Ollama 将 LoRA 合并后部署")


if __name__ == "__main__":
    main()
