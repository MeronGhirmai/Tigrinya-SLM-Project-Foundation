"""
src/publish_to_hf.py
Publish the trained Tigrinya Gemma-2-2B QLoRA adapter to the Hugging Face Hub
and/or Kaggle Models.

Run this on the machine that trained the model (Colab / Kaggle), after
`src/train_qlora.py` (and optionally `src/evaluate_model.py`) has finished.

Credentials
  Hugging Face : `huggingface-cli login`, `notebook_login()`, or `export HF_TOKEN=hf_...`
                 (the token needs WRITE permission)
  Kaggle       : `export KAGGLE_USERNAME=... KAGGLE_KEY=...` or ~/.kaggle/kaggle.json
                 (create an API token at https://www.kaggle.com/settings)

Examples
  python3 src/publish_to_hf.py --hf_repo_id USER/gemma2-2b-tigrinya-qlora
  python3 src/publish_to_hf.py --kaggle_handle USER/gemma2-2b-tigrinya-qlora/transformers/default
  python3 src/publish_to_hf.py --hf_repo_id ... --kaggle_handle ...   # both
"""
import argparse
import json
import os
import struct
import sys

# Files the trainer writes that should never be published (optimizer state, RNG, etc.).
IGNORE_PATTERNS = [
    "checkpoint-*",
    "checkpoint-*/**",
    "runs/**",
    "*.pt",
    "*.pth",
    "training_args.bin",
    "rng_state*",
    "human_eval_sheet.csv",
]
REQUIRED_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def count_safetensors_params(path: str) -> int:
    """Count parameters by reading only the safetensors JSON header (no torch needed)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for dim in meta["shape"]:
            n *= dim
        total += n
    return total


def load_json(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def build_model_card(adapter_dir: str, hf_repo_id: str, kaggle_handle: str) -> str:
    cfg = load_json(os.path.join(adapter_dir, "adapter_config.json")) or {}
    base_id = cfg.get("base_model_name_or_path", "google/gemma-2-2b")
    r, alpha = cfg.get("r"), cfg.get("lora_alpha")
    targets = cfg.get("target_modules") or []
    if isinstance(targets, str):
        targets = [targets]
    n_params = count_safetensors_params(os.path.join(adapter_dir, "adapter_model.safetensors"))

    metrics = load_json(os.path.join(adapter_dir, "eval_results.json"))
    if metrics:
        metric_lines = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
        eval_section = f"| Metric | Value |\n|---|---|\n{metric_lines}\n"
    else:
        eval_section = "_Evaluation results not yet published. Run `src/evaluate_model.py` and re-publish._\n"

    load_id = hf_repo_id or "<path-or-repo-of-this-adapter>"
    links = []
    if hf_repo_id:
        links.append(f"- Hugging Face: https://huggingface.co/{hf_repo_id}")
    if kaggle_handle:
        links.append(f"- Kaggle: https://www.kaggle.com/models/{'/'.join(kaggle_handle.split('/')[:2])}")
    links.append("- Code: https://github.com/MeronGhirmai/Tigrinya-SLM-Project-Foundation")

    return f"""---
language:
- ti
license: gemma
base_model: {base_id}
library_name: peft
pipeline_tag: text-generation
tags:
- tigrinya
- geez
- low-resource-nlp
- qlora
- lora
- gemma-2
---

# Gemma-2-2B Tigrinya (ትግርኛ) — QLoRA Adapter

LoRA adapter weights for [`{base_id}`](https://huggingface.co/{base_id}), fine-tuned with 4-bit QLoRA
on Tigrinya text written in the Ge'ez (ፊደል) script. Built as a Google DeepMind AI Research
Foundations capstone project.

This repository contains **only the adapter**. You also need access to the gated base model
(accept the Gemma license at https://huggingface.co/{base_id}).

{chr(10).join(links)}

## Training setup
- **Base model:** `{base_id}` (loaded in 4-bit NF4, double quantization)
- **LoRA:** r={r}, alpha={alpha}, dropout={cfg.get("lora_dropout")}
- **Target modules:** {", ".join(f"`{t}`" for t in sorted(targets))}
- **Adapter parameters:** {n_params:,}
- **Optimizer:** paged AdamW 8-bit, lr 2e-4, cosine schedule, max sequence length 512

## Evaluation
{eval_section}
## Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_id = "{base_id}"
adapter_id = "{load_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto", attn_implementation="eager")
model = PeftModel.from_pretrained(base, adapter_id)

prompt = "<start_of_turn>user\\nተምሃሮ ብቋንቋ ኣዲኦም ክምሃሩ ከለዉ ዘለዎ ጥቕሚ ግለጽ።<end_of_turn>\\n<start_of_turn>model\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.15)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Intended use & limitations
- Research and educational use for Tigrinya language generation.
- Covers both Eritrean and Ethiopian Tigrinya registers; quality varies by dialect and domain.
- Small model, small dataset: outputs can be fluent but factually wrong. Not for medical, legal,
  or other high-stakes use without human review.

## License
These adapter weights are a Model Derivative of Gemma and are distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Use is also subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
"""


def publish_hf(adapter_dir: str, repo_id: str, token, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"[HF] Creating/verifying repo {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[HF] Uploading {adapter_dir} ...")
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload Tigrinya Gemma-2-2B QLoRA adapter and model card",
    )
    print(f"[HF] Done: https://huggingface.co/{repo_id}")


def publish_kaggle(adapter_dir: str, handle: str, license_name: str, notes: str) -> None:
    import kagglehub

    print(f"[Kaggle] Uploading {adapter_dir} to {handle} ...")
    kagglehub.model_upload(
        handle=handle,
        local_model_dir=adapter_dir,
        license_name=license_name,
        version_notes=notes,
        ignore_patterns=IGNORE_PATTERNS,
    )
    owner, slug = handle.split("/")[:2]
    print(f"[Kaggle] Done: https://www.kaggle.com/models/{owner}/{slug}")


def main():
    p = argparse.ArgumentParser(description="Publish the Tigrinya QLoRA adapter to Hugging Face and/or Kaggle")
    p.add_argument("--adapter_dir", default="outputs/gemma2-2b-tigrinya-qlora")
    p.add_argument("--hf_repo_id", help="e.g. YourHFUser/gemma2-2b-tigrinya-qlora")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF write token (defaults to $HF_TOKEN, then the cached login)")
    p.add_argument("--private", action="store_true", help="Create the HF repo as private")
    p.add_argument("--kaggle_handle", help="e.g. yourkaggleuser/gemma2-2b-tigrinya-qlora/transformers/default")
    p.add_argument("--kaggle_license", default="Apache 2.0",
                   help="Kaggle license name, applied only when the model is first created "
                        "(the uploaded README states the Gemma Terms of Use apply)")
    p.add_argument("--version_notes", default="QLoRA adapter for Tigrinya")
    args = p.parse_args()

    if not args.hf_repo_id and not args.kaggle_handle:
        p.error("pass --hf_repo_id and/or --kaggle_handle")
    if args.kaggle_handle and len(args.kaggle_handle.split("/")) != 4:
        p.error("--kaggle_handle must look like USER/MODEL_SLUG/FRAMEWORK/VARIATION, "
                "e.g. meronghirmai/gemma2-2b-tigrinya-qlora/transformers/default")

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(args.adapter_dir, f))]
    if missing:
        sys.exit(f"Adapter not found in {args.adapter_dir} (missing {missing}). "
                 "Run src/train_qlora.py first, on the same machine you publish from.")

    readme = os.path.join(args.adapter_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(build_model_card(args.adapter_dir, args.hf_repo_id, args.kaggle_handle))
    print(f"Wrote model card: {readme}")

    if args.hf_repo_id:
        publish_hf(args.adapter_dir, args.hf_repo_id, args.token, args.private)
    if args.kaggle_handle:
        publish_kaggle(args.adapter_dir, args.kaggle_handle, args.kaggle_license, args.version_notes)


if __name__ == "__main__":
    main()
