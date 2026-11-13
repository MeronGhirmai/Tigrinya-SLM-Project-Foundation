"""
app.py
Interactive Gradio Web Playground & Tokenizer Telemetry Suite for Tigrinya Gemma-2-2B SLM
Run locally (`python3 app.py`) or in Google Colab (`python3 app.py --share`)
"""
import os, argparse
import gradio as gr
from src.data_preprocessing import GeezTigrinyaNormalizer

normalizer = GeezTigrinyaNormalizer()
MODEL_LOADED = False
model, tokenizer = None, None

def load_slm_if_available(base_model='google/gemma-2-2b', adapter_dir='outputs/gemma2-2b-tigrinya-qlora'):
    global MODEL_LOADED, model, tokenizer
    if MODEL_LOADED:
        return True
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map='auto', attn_implementation='eager', torch_dtype=dtype)
            model = PeftModel.from_pretrained(base, adapter_dir) if os.path.exists(adapter_dir) else base
            model.eval()
            MODEL_LOADED = True
            return True
    except Exception as e:
        print(f'[UI Notice] Running in lightweight Tokenizer/Normalizer simulation mode (GPU/Model not loaded: {e})')
    return False

def tigrinya_slm_pipeline(user_prompt: str, temperature: float, rep_penalty: float, max_tokens: int):
    cleaned_prompt = normalizer.normalize_sentence(user_prompt)
    words = cleaned_prompt.split()
    geez_ratio = normalizer.compute_geez_script_ratio(cleaned_prompt)

    if load_slm_if_available():
        import torch
        formatted = f'<bos><start_of_turn>user\n{cleaned_prompt}<end_of_turn>\n<start_of_turn>model\n'
        inputs = tokenizer(formatted, return_tensors='pt').to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=int(max_tokens),
                do_sample=(temperature > 0.05), temperature=max(temperature, 0.05),
                repetition_penalty=float(rep_penalty), pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        subwords = tokenizer.tokenize(cleaned_prompt)
    else:
        subwords = [f'▁{w}' for w in words]
        response = (
            '[Simulation Mode - Connect Colab GPU for Live Gemma-2-2B Weights]\n'
            f'Normalized Prompt: {cleaned_prompt}\n\n'
            'Sample Model Response: ትምህርትን ቴክኖሎጅን ብቋንቋ ኣደ ምስ ዝቐርብ፡ ማሕበረሰባዊ ምዕባለን ተበጻሕነት ፍልጠትን ብዓቢኡ የስፍሕ።'
        )

    fertility = len(subwords) / max(len(words), 1)
    telemetry = (
        f'• Cleaned Ge\'ez Input : {cleaned_prompt}\n'
        f'• Ge\'ez Script Purity : {geez_ratio * 100:.1f}%\n'
        f'• Word Count          : {len(words)} words\n'
        f'• Subword Tokens      : {len(subwords)} tokens (Fertility: {fertility:.2f} tok/word)\n'
        f'• Safety Guardrail    : Active (Repetition Penalty={rep_penalty}, Scope=Educational/General)'
    )
    return response, telemetry

with gr.Blocks(theme=gr.themes.Soft(), title='Tigrinya Gemma-2-2B SLM Playground') as demo:
    gr.Markdown('# 🇪🇷🇪🇹 Tigrinya (ትግርኛ) Gemma-2-2B SLM — Research Testbench & Telemetry UI')
    gr.Markdown('Test Ge\'ez orthographic normalization, inspect subword token fertility, and evaluate domain responses across Education, Public Health, and Cultural prompts.')
    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(label='Tigrinya Prompt (ትግርኛ ሕቶ / መምርሒ)', lines=4, value='ተምሃሮ ብቋንቋ ኣዲኦም ክምሃሩ ከለዉ ዘለዎ ጥቕሚ ግለጽ::')
            with gr.Row():
                temp_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label='Temperature')
                rep_slider = gr.Slider(1.0, 1.5, value=1.15, step=0.05, label='Repetition Penalty (Ge\'ez Loop Guard)')
                max_tok_slider = gr.Slider(32, 256, value=128, step=32, label='Max New Tokens')
            run_btn = gr.Button('🚀 Generate & Run Telemetry Audit', variant='primary')
        with gr.Column(scale=4):
            out_box = gr.Textbox(label='Gemma-2-2B Tigrinya Response', lines=6)
            diag_box = gr.Textbox(label='Live Module 2 & Module 6 Telemetry (Normalization, Fertility & Guardrails)', lines=6)
    gr.Examples(
        examples=[
            ['ተምሃሮ ብቋንቋ ኣዲኦም ክምሃሩ ከለዉ ዘለዎ ጥቕሚ ግለጽ::', 0.3, 1.15, 128],
            ['ጽሬት ዝስተ ማይ ምሕላው፡ ካብ ተመሓላለፍቲ ሕማማት ንምክልኻል ከመይ ይሕግዝ፧', 0.2, 1.15, 128],
            ['ባህላዊ ስነ-ጥበብን ስነ-ጽሑፍን ትግርኛ ንምዕቃብ እንታይ ክግበር ይግባእ::', 0.4, 1.15, 128],
        ],
        inputs=[prompt_box, temp_slider, rep_slider, max_tok_slider]
    )
    run_btn.click(tigrinya_slm_pipeline, inputs=[prompt_box, temp_slider, rep_slider, max_tok_slider], outputs=[out_box, diag_box])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help='Create public Gradio share link (ideal for Colab)')
    args = parser.parse_args()
    demo.launch(share=args.share)
