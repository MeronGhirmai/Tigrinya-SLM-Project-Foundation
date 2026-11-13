import os, json, math, argparse, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sacrebleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='google/gemma-2-2b')
    parser.add_argument('--adapter_dir', type=str, default='outputs/gemma2-2b-tigrinya-qlora')
    parser.add_argument('--test_file', type=str, default='data/processed/test.jsonl')
    parser.add_argument('--output_eval_csv', type=str, default='outputs/human_eval_sheet.csv')
    args = parser.parse_args()

    with open(args.test_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (7, 5)
    compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map='auto', attn_implementation='eager', torch_dtype=compute_dtype)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir) if os.path.exists(args.adapter_dir) else base_model
    model.eval()

    total_nll, total_tokens, total_words = 0.0, 0, 0
    hypotheses, references, human_rows = [], [], []
    with torch.no_grad():
        for idx, rec in enumerate(records):
            inputs = tokenizer(rec['text'], return_tensors='pt', truncation=True, max_length=512).to(model.device)
            seq_len = inputs['input_ids'].size(1)
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            total_nll += outputs.loss.item() * (seq_len - 1)
            total_tokens += (seq_len - 1)
            total_words += max(len(rec.get('response', rec['text']).split()), 1)
            prompt_str = f"<bos><start_of_turn>user\n{rec['instruction']}<end_of_turn>\n<start_of_turn>model\n"
            p_inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)
            gen_ids = model.generate(**p_inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id)
            pred = tokenizer.decode(gen_ids[0][p_inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            hypotheses.append(pred)
            references.append(rec['response'])
            human_rows.append({'Sample_ID': idx + 1, 'Prompt_Tigrinya': rec['instruction'], 'Reference_Gold': rec['response'], 'Model_Prediction': pred})

    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
    print(f'Token PPL: {math.exp(min(total_nll / max(total_tokens, 1), 20.0)):.3f} | ChrF++: {chrf:.2f}')
    os.makedirs(os.path.dirname(args.output_eval_csv), exist_ok=True)
    pd.DataFrame(human_rows).to_csv(args.output_eval_csv, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()
