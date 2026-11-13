"""
tests/test_pipeline.py
Automated Unit Tests for Tigrinya Ge'ez Orthographic Normalization & Data Integrity
Run with: `python3 tests/test_pipeline.py`
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import GeezTigrinyaNormalizer

class TestTigrinyaSLMPipeline(unittest.TestCase):
    def setUp(self):
        self.norm = GeezTigrinyaNormalizer(replace_wordspace_with_space=True, min_geez_ratio=0.60)

    def test_full_stop_disambiguation(self):
        raw = 'ትግርኛ ሴማዊ ቋንቋ እዩ:: ምዕባለ ይፈጥር፡፡'
        cleaned = self.norm.normalize_sentence(raw)
        self.assertIn('እዩ።', cleaned)
        self.assertIn('ይፈጥር።', cleaned)
        self.assertNotIn('::', cleaned)
        self.assertNotIn('፡፡', cleaned)

    def test_phonemic_pharyngeal_preservation(self):
        # Ensure Tigrinya phonemic ሐ/ሀ and ዐ/አ are NEVER collapsed into Amharic homophones
        raw = 'ሓዳስ ዓመት ኣብ ኤርትራን ኢትዮጵያን።'
        cleaned = self.norm.normalize_sentence(raw)
        self.assertIn('ሓዳስ', cleaned)
        self.assertIn('ዓመት', cleaned)

    def test_wordspace_normalization(self):
        raw = 'ምዕባለ፡ሰብኣዊ፡ኣስተውዕሎ።'
        cleaned = self.norm.normalize_sentence(raw)
        self.assertEqual(cleaned, 'ምዕባለ ሰብኣዊ ኣስተውዕሎ።')

if __name__ == '__main__':
    print('Running Tigrinya SLM Automated Unit Tests...')
    unittest.main(verbosity=2)
