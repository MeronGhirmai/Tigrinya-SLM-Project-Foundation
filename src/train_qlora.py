import os, argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/gemma-2-2b')
    parser.add_argument('--train_file', type=str, default='data/processed/train.jsonl')
    parser.add_argument('--val_file', type=str, default='data/processed/val.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs/gemma2-2b-tigrinya-qlora')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (7, 5)
    compute_dtype, use_bf16, use_fp16 = (torch.bfloat16, True, False) if major >= 8 else (torch.float16, False, True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, device_map='auto', attn_implementation='eager', torch_dtype=compute_dtype)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, peft_config)

    dataset = load_dataset('json', data_files={'train': args.train_file, 'validation': args.val_file})
    sft_config = SFTConfig(
        output_dir=args.output_dir, dataset_text_field='text', max_seq_length=512,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, optim='paged_adamw_8bit', learning_rate=2e-4,
        lr_scheduler_type='cosine', warmup_ratio=0.05, weight_decay=0.01,
        num_train_epochs=args.epochs, bf16=use_bf16, fp16=use_fp16, logging_steps=5, report_to='none'
    )
    trainer = SFTTrainer(model=model, train_dataset=dataset['train'], eval_dataset=dataset['validation'], peft_config=peft_config, processing_class=tokenizer, args=sft_config)
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
