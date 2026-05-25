"""
LoRA 权重合并脚本
将 SFT → GRPO → DPO 三阶段的增量 LoRA 合并回基底模型，输出完整权重。

说明：
  - 合并在 CPU fp16 下进行，不占 GPU 显存
  - 最终模型保存为 safetensors，可直接被 transformers 或 llama.cpp 加载
  - 若只跑了部分阶段，--adapter 指向最后一阶段的输出目录即可

用法：
    python train/merge.py
    python train/merge.py --base models/Qwen2.5-7B-Instruct --adapter output/dpo --output output/merged
"""
import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_path: str, adapter_path: str, output_path: str):
    print(f"[Merge] 基底模型  : {base_path}")
    print(f"[Merge] LoRA 适配器: {adapter_path}")
    print(f"[Merge] 输出路径  : {output_path}")
    print(f"[Merge] 加载模式  : CPU fp16（不占 GPU 显存）\n")

    t0 = time.time()

    print("① 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    print("② 加载基底模型（fp16，CPU）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("③ 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("④ 合并权重...")
    model = model.merge_and_unload()

    print(f"⑤ 保存合并模型 → {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    elapsed = time.time() - t0
    print(f"\n[Merge] 完成，耗时 {elapsed:.0f}s")
    print(f"[Merge] 合并模型已保存至 {output_path}")
    print("[Merge] 后续选项:")
    print("         A. 直接推理：python scripts/eval_compare.py --ft-model output/merged")
    print("         B. 转 GGUF：llama.cpp/convert_hf_to_gguf.py output/merged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="output/dpo",
                        help="最后一阶段输出目录（SFT/GRPO/DPO 任选其一）")
    parser.add_argument("--output",  default="output/merged")
    args = parser.parse_args()

    merge(args.base, args.adapter, args.output)


if __name__ == "__main__":
    main()
