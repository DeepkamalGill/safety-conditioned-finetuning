import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with SFTTrainer")

    # Model & data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data_path", type=str, default="data/generated/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen2.5-1.5b-safety-lora")

    # Training hyperparameters
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"]
    )

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    # LOAD TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # LOAD DATA
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{args.data_path}/train.jsonl",
            "eval": f"{args.data_path}/eval.jsonl"
        }
    )

    # LOAD MODEL (BF16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False

    # LoRA CONFIG
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # TRAINING ARGS
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False
    )

    # FORMAT FUNCTION
    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            max_length=args.max_seq_len,
            tokenize=False
        )

    # TRAINER
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        formatting_func=format_chat,
        args=training_args
    )

    # TRAIN
    trainer.train()

    # SAVE
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
