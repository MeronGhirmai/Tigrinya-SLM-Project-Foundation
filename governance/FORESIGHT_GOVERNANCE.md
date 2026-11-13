# Module 6: Foresight Governance Blueprint

## Anticipatory Failure Modes & Mitigations
1. **Amharic Cross-Lingual Code-Bleed:** Base LLMs see 10x more Amharic in the Ge'ez Unicode block. Mitigated via `min_geez_ratio=0.60` filtering, curated Tigrinya SFT splits, and native-speaker rubric auditing.
2. **Epistemic Over-Reliance & Medical/Legal Hallucinations:** A 2.6B parameter model has bounded world knowledge. Autonomous clinical/legal deployment is prohibited in the Model Card and enforced via UI disclaimers.
3. **Geopolitical & Conflict Polarization:** Mitigated by prioritizing HornMT health, agricultural, and educational domains over unmoderated political forums.
4. **Ge'ez Syllable Repetition Loops:** High token fertility (~4.0 tok/word) can trigger greedy decoding loops; mitigated via `repetition_penalty=1.15` in `src/evaluate_model.py` and `app.py`.
