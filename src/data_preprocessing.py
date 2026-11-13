import os, re, json, argparse, unicodedata
from pathlib import Path
from typing import List, Dict
import pandas as pd
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

class GeezTigrinyaNormalizer:
    GEEZ_CHAR_PATTERN = re.compile(r'[\u1200-\u139F]')
    def __init__(self, replace_wordspace_with_space: bool = True, min_geez_ratio: float = 0.60):
        self.replace_wordspace_with_space = replace_wordspace_with_space
        self.min_geez_ratio = min_geez_ratio

    def normalize_sentence(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip(): return ''
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'https?://\S+|www\.\S+|<[^>]+>', '', text)
        text = re.sub(r'(?::\s*:|፡\s*፡|:\s*፡|፡\s*:)', '።', text)
        if self.replace_wordspace_with_space:
            text = re.sub(r'፡', ' ', text)
        text = re.sub(r'\s*([\u1362-\u1368])\s*', r'\1 ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def compute_geez_script_ratio(self, text: str) -> float:
        non_ws = re.sub(r'\s+', '', text)
        return len(self.GEEZ_CHAR_PATTERN.findall(non_ws)) / len(non_ws) if non_ws else 0.0

    def clean_corpus_dataframe(self, df: pd.DataFrame, text_col: str = 'text', min_words: int = 4, max_words: int = 512) -> pd.DataFrame:
        df = df.copy()
        df['cleaned_text'] = df[text_col].astype(str).apply(self.normalize_sentence)
        df['geez_ratio'] = df['cleaned_text'].apply(self.compute_geez_script_ratio)
        df['word_count'] = df['cleaned_text'].apply(lambda s: len(s.split()))
        df = df[(df['geez_ratio'] >= self.min_geez_ratio) & (df['word_count'] >= min_words) & (df['word_count'] <= max_words)]
        return df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)

def evaluate_tokenizer_efficiency(tokenizer, texts: List[str], tokenizer_label: str = 'google/gemma-2-2b') -> Dict[str, float]:
    total_words, total_chars, total_tokens, byte_fallback_count = 0, 0, 0, 0
    for text in texts:
        total_words += len(text.split())
        total_chars += len(text.replace(' ', ''))
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids if hasattr(encoded, 'ids') else tokenizer.encode(text, add_special_tokens=False)
        tokens = encoded.tokens if hasattr(encoded, 'tokens') else tokenizer.convert_ids_to_tokens(token_ids)
        total_tokens += len(token_ids)
        for tok in tokens:
            if isinstance(tok, str) and re.match(r'^<0x[0-9A-Fa-f]{2}>$', tok):
                byte_fallback_count += 1
    return {
        'Tokenizer': tokenizer_label,
        'Words': total_words,
        'Tokens': total_tokens,
        'Fertility (Tok/Word)': round(total_tokens / max(total_words, 1), 3),
        'Chars/Token': round(total_chars / max(total_tokens, 1), 3),
        'Byte-Fallback (%)': round((byte_fallback_count / max(total_tokens, 1)) * 100.0, 2),
    }

def train_candidate_tigrinya_bpe(texts: List[str], vocab_size: int = 500) -> Tokenizer:
    bpe_tokenizer = Tokenizer(models.BPE(unk_token='<unk>', byte_fallback=True))
    bpe_tokenizer.normalizer = normalizers.NFC()
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement='▁')
    bpe_tokenizer.decoder = decoders.Metaspace(replacement='▁')
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, special_tokens=['<pad>', '<eos>', '<bos>', '<unk>'])
    bpe_tokenizer.train_from_iterator(texts, trainer=trainer)
    return bpe_tokenizer

def export_gemma_instruction_jsonl(df: pd.DataFrame, output_dir: str = 'data/processed') -> None:
    os.makedirs(output_dir, exist_ok=True)
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df_shuffled)
    t_end, v_end = max(1, int(n * 0.70)), max(2, int(n * 0.85))
    splits = {'train': df_shuffled.iloc[:t_end], 'val': df_shuffled.iloc[t_end:v_end], 'test': df_shuffled.iloc[v_end:]}
    for split_name, split_df in splits.items():
        if len(split_df) == 0: split_df = df_shuffled.iloc[:1]
        out_path = Path(output_dir) / f'{split_name}.jsonl'
        with open(out_path, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                prompt = row.get('instruction', 'ነዚ ዝስዕብ ሓሳብ ብጽፉፍን ግቡእን ትግርኛ ገሊጽካ ጸሓፎ፦')
                response = row['cleaned_text']
                text = f'<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn><eos>'
                f.write(json.dumps({'instruction': prompt, 'response': response, 'text': text, 'word_count': int(row['word_count'])}, ensure_ascii=False) + '\n')
        print(f'[Export] Saved {len(split_df)} records -> {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/gemma-2-2b')
    parser.add_argument('--input_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    args = parser.parse_args()
    seed_corpus = [
        'ትግርኛ ኣብ ኤርትራን ሰሜን ኢትዮጵያን ብሰፊሑ ዝዝረብ ሴማዊ ቋንቋ እዩ::',
        'ምዕባለ ሰብኣዊ ኣስተውዕሎ፡ ነቶም ውሑድ ዲጂታላዊ ጸጋታት ዘለዎም ቋንቋታት ዓቢ ዕድል ይፈጥር፡፡',
        'ተምሃሮ ብቋንቋ ኣዲኦም ክምሃሩ ከለዉ፡ ፍልጠት ብቐሊሉ ንምርዳእን ምምርማርን ይሕግዞም።',
        'ባህላዊ ስነ-ጥበብን ስነ-ጽሑፍን ትግርኛ፡ ንወለዶታት ዝሰገረ ታሪኻዊ ውርሻ ኣለዎ፡ ስለዚ ክዕቀብ ይግባእ።',
        'ሕክምናዊ ሓበሬታን ትምህርታዊ ጽሑፋትን ብፊደል ግዕዝ ብምድላው፡ ተበጻሕነት ቴክኖሎጂ ነስፍሕ።',
    ]
    df_raw = pd.read_csv(args.input_csv) if (args.input_csv and os.path.exists(args.input_csv)) else pd.DataFrame({'text': seed_corpus})
    df_clean = GeezTigrinyaNormalizer().clean_corpus_dataframe(df_raw, text_col='text')
    cleaned_sentences = df_clean['cleaned_text'].tolist()
    rows = []
    try:
        gemma_tok = AutoTokenizer.from_pretrained(args.model_id)
        rows.append(evaluate_tokenizer_efficiency(gemma_tok, cleaned_sentences, f'{args.model_id} (256k)'))
    except Exception as exc:
        print(f'[Notice] HF Hub auth needed for {args.model_id}: {exc}')
    custom_bpe = train_candidate_tigrinya_bpe(cleaned_sentences, vocab_size=500)
    rows.append(evaluate_tokenizer_efficiency(custom_bpe, cleaned_sentences, 'Custom Tigrinya BPE'))
    print(pd.DataFrame(rows).to_string(index=False))
    export_gemma_instruction_jsonl(df_clean, output_dir=args.output_dir)
