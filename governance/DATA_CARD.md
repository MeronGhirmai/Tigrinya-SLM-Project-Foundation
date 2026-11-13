# Data Card: Tigrinya (ትግርኛ) Multi-Domain Instruction & Language Modeling Corpus

## 1. Dataset Identity & Provenance
- **Target Language & Script:** Tigrinya (`tir`, ISO 639-3) written in the Ge'ez Abugida (`Ethi`, ISO 15924, Unicode `U+1200`–`U+139F`).
- **Curated Upstream Sources:** HornMT (Hadgu et al.), XL-Sum BBC Tigrinya, MasakhaNEWS (Tigrinya), and TLMD.
- **Licensing & Data Sovereignty:** Verified CC-BY-4.0 / CC-BY-NC-4.0 open benchmarks.

## 2. Socio-Linguistic & Orthographic Design Decisions (`src/data_preprocessing.py`)
- **Phonemic Fidelity Over Amharic-Centric Simplification:** Preserves pharyngeal fricatives/stops (`ሐ`, `ዐ`) vs glottal counterparts (`ሀ`, `አ`) critical to Tigrinya lexical meaning.
- **Punctuation Standardization:** Canonicalizes `::` and `፡፡` into Ge'ez Full Stop (`።` `U+1362`) and converts archaic wordspaces (`፡` `U+1361`) to standard space.
- **Script Purity Filtering (`min_geez_ratio=0.60`):** Removes web noise while allowing bilingual educational context.

## 3. Known Representation Skew & Societal Risks
- **Register Skew:** Digital corpora skew toward formal urban journalism (Asmara and Mekelle standards); spoken regional varieties (Raya, Adwa, Shire, Agame, Seraye) are under-represented.
- **Cross-Lingual Bleed Risk:** Mitigates Amharic lexical intrusion (`ነው` vs `እዩ`) via strict corpus curation and human evaluation rubrics.
