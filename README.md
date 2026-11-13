# 🇪🇷🇪🇹 Tigrinya (ትግርኛ) Gemma-2-2B SLM — Google DeepMind Capstone

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeronGhirmai/Tigrinya-SLM-Project-Foundation/blob/main/notebooks/Tigrinya_Gemma2_Colab_Master.ipynb)
![Base Model](https://img.shields.io/badge/Base_Model-google%2Fgemma--2--2b-blue)
![Language](https://img.shields.io/badge/Language-Tigrinya_(ትግርኛ)-green)
![VRAM](https://img.shields.io/badge/Peak_VRAM-~7.61_GB_(T4%2FL4)-orange)

An end-to-end, single-GPU memory-optimized pipeline for curating, fine-tuning, evaluating, and governing a **Gemma 2 (2B)** Small Language Model for **Tigrinya (ትግርኛ)** written in the **Ge'ez (ፊደል)** abugida script.

## 🌟 Interactive Portal & Quick Links
- **Interactive Web Portal:** Open `index.html` in any browser to explore the architecture dashboard and live Ge'ez Normalizer sandbox.
- **1-Click Google Colab Notebook:** `notebooks/Tigrinya_Gemma2_Colab_Master.ipynb`
- **Live Gradio Playground UI:** Run `python3 app.py` (or `python3 app.py --share` on Colab).

## 🧪 Professional 4-Step Verification & Test Workflow
1. **Run Automated Unit Tests (CPU):** `python3 tests/test_pipeline.py`
2. **Run Ge'ez Normalization & Tokenizer Fertility Audit:** `python3 src/data_preprocessing.py --model_id google/gemma-2-2b`
3. **Run Single-GPU 4-Bit QLoRA Fine-Tuning (T4/L4 GPU):** `python3 src/train_qlora.py --model_id google/gemma-2-2b --batch_size 2 --grad_accum 8 --epochs 3`
4. **Run Quantitative (`PPL`, `ChrF++`) & Native Speaker Rubric Export:** `python3 src/evaluate_model.py --base_model google/gemma-2-2b --adapter_dir outputs/gemma2-2b-tigrinya-qlora`
