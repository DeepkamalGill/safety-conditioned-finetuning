import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/qwen2.5-1.5b-safety-lora-fp16",
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/qwen2.5-1.5b-safety-merged",
        help="Directory to save merged model"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapters...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging LoRA adapters into base model...")
    merged = peft_model.merge_and_unload()

    print("Saving merged model...")
    merged.save_pretrained(args.output_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
