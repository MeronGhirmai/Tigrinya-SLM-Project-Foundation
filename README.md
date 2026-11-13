# Tigrinya SLM Project Foundation - Google DeepMind Capstone

End-to-end single-GPU QLoRA fine-tuning, Ge'ez tokenization diagnostics, and Responsible AI governance for Tigrinya (google/gemma-2-2b).

## Quickstart Commands
1. pip install -r requirements.txt
2. python src/data_preprocessing.py --model_id google/gemma-2-2b
3. python src/train_qlora.py --model_id google/gemma-2-2b --batch_size 2 --grad_accum 8 --epochs 3
4. python src/evaluate_model.py --base_model google/gemma-2-2b --adapter_dir outputs/gemma2-2b-tigrinya-qlora
