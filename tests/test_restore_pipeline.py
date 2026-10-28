import tempfile
import unittest

from train_edit_tagger import (
    COPY_LABEL,
    align_edit_outputs,
    label_to_output,
    output_to_label,
)
from collections import Counter
from pathlib import Path

from train_contextual_corrector import apply_contextual_mapping, tuning_key
from infer_restore import configure_utf8_stdout
from evaluate_restore import compute_metrics
from prepare_segment_data import extract_trusted_segments, split_restore_segments
from tibetan_syllable import (
    apply_syllable_vocabulary_correction,
    build_syllable_vocabulary,
    is_likely_tibetan_syllable,
    split_syllable_units,
)
from train_transducer import apply_mapping, learn_char_mapping
from train_restore import CharTokenizer, build_vocab, read_parallel_data, split_pairs


class RestorePipelineTests(unittest.TestCase):
    def test_edit_alignment_reconstructs_reference(self):
        source = "རྣཾར"
        reference = "རྣམ་ཐར"
        outputs = align_edit_outputs(source, reference)
        self.assertEqual(len(outputs), len(source))
        self.assertEqual("".join(outputs), reference)

    def test_copy_label_round_trip(self):
        self.assertEqual(output_to_label("ར", "ར"), COPY_LABEL)
        self.assertEqual(label_to_output("ར", COPY_LABEL), "ར")

    def test_inference_stdout_is_reconfigured_to_utf8_on_windows(self):
        class FakeStdout:
            def __init__(self):
                self.encoding = "gbk"

            def reconfigure(self, encoding):
                self.encoding = encoding

        stream = FakeStdout()
        configure_utf8_stdout(stream)

        self.assertEqual(stream.encoding, "utf-8")

    def test_read_parallel_data_keeps_pairs_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Source.txt"
            restore = root / "Restore.txt"
            source.write_text("\ufeffཀྕ་\nཀ་༡་\n", encoding="utf-8")
            restore.write_text("\ufeffཀ་ཅ།\nཀ་གཅིག།\n", encoding="utf-8")

            self.assertEqual(
                read_parallel_data(source, restore),
                [("ཀྕ་", "ཀ་ཅ།"), ("ཀ་༡་", "ཀ་གཅིག།")],
            )

            restore.write_text("ཀ་ཅ།\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_parallel_data(source, restore)

    def test_build_vocab_adds_special_tokens_and_data_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_file = Path(tmp) / "tibetan_VCC_7387.txt"
            vocab_file.write_text("ཀ\n་\n།\nཀྱོ\n", encoding="utf-8")

            stoi, itos = build_vocab(vocab_file, [("ཀ s", "ཀ.")])

        for token in ["<pad>", "<bos>", "<eos>", "<unk>", "ཀ", " ", "s", "."]:
            self.assertIn(token, stoi)
            self.assertEqual(itos[stoi[token]], token)
        self.assertNotIn("ཀྱོ", stoi)

    def test_tokenizer_round_trip_ignores_special_tokens(self):
        stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "ཀ": 4, "་": 5}
        tokenizer = CharTokenizer(stoi)

        ids = tokenizer.encode("ཀ་", add_bos=True, add_eos=True)

        self.assertEqual(ids, [1, 4, 5, 2])
        self.assertEqual(tokenizer.decode(ids), "ཀ་")

    def test_split_pairs_is_reproducible_and_exhaustive(self):
        pairs = [(str(i), str(i)) for i in range(20)]

        train, valid, test = split_pairs(pairs, valid_ratio=0.2, test_ratio=0.1, seed=7)
        train2, valid2, test2 = split_pairs(pairs, valid_ratio=0.2, test_ratio=0.1, seed=7)

        self.assertEqual((train, valid, test), (train2, valid2, test2))
        self.assertEqual(len(train) + len(valid) + len(test), len(pairs))
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 2)

    def test_split_restore_segments_keeps_tibetan_shad(self):
        self.assertEqual(
            split_restore_segments("ཀ་ཅ། ཀ་གཅིག་མ།"),
            ["ཀ་ཅ།", "ཀ་གཅིག་མ།"],
        )
        self.assertEqual(split_restore_segments("ཀ་ལ་པིང་ཀ"), ["ཀ་ལ་པིང་ཀ"])

    def test_extract_trusted_segments_only_uses_matching_counts(self):
        pairs = [
            ("ཀྕ་  ཀ་༡་མ", "ཀ་ཅ། ཀ་གཅིག་མ།"),
            ("ཀྕ་ ཀ་༡་མ", "ཀ་ཅ། ཀ་གཅིག་མ།"),
            ("ཀརྨ་དེིནེཊ་པ་  དེ་༤ནཊེ་པ་", "ཀརྨ་དེ་བཞིན་བཤེགས་པ།"),
            ("ཀནྷོ་བོད་རིཊ་", "ཀན་ལྷོ་བོད་རིགས།"),
        ]

        segments, stats = extract_trusted_segments(pairs)

        self.assertEqual(
            segments,
            [
                ("ཀྕ་", "ཀ་ཅ།"),
                ("ཀ་༡་མ", "ཀ་གཅིག་མ།"),
                ("ཀནྷོ་བོད་རིཊ་", "ཀན་ལྷོ་བོད་རིགས།"),
            ],
        )
        self.assertEqual(stats["accepted_lines"], 2)
        self.assertEqual(stats["rejected_lines"], 2)

    def test_compute_metrics_reports_exact_and_edit_errors(self):
        rows = [
            {"source": "a", "reference": "abc", "prediction": "abc"},
            {"source": "x", "reference": "abc", "prediction": "adc"},
        ]

        metrics = compute_metrics(rows)

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["line_error_rate"], 0.5)
        self.assertEqual(metrics["edit_substitutions"], 1)
        self.assertEqual(metrics["edit_insertions"], 0)
        self.assertEqual(metrics["edit_deletions"], 0)
        self.assertAlmostEqual(metrics["micro_cer"], 1 / 6)

    def test_char_transducer_learns_explainable_expansions(self):
        pairs = [
            ("a1", "aONE"),
            ("b1", "bONE"),
            ("c1", "cONE"),
        ]

        mapping, _ = learn_char_mapping(pairs, iterations=2, max_output_length=8)

        self.assertEqual(mapping["1"], "ONE")
        self.assertEqual(apply_mapping("b1", mapping), "bONE")

    def test_contextual_mapping_uses_neighbor_specific_output(self):
        base_mapping = {"a": "a", "x": "X", "b": "b", "c": "c"}
        context_counts = {
            ("tri", "a", "x", "b"): Counter({"LEFT": 3}),
            ("tri", "a", "x", "c"): Counter({"RIGHT": 3}),
        }

        self.assertEqual(
            apply_contextual_mapping(
                "axb",
                base_mapping,
                context_counts,
                min_count=2,
                min_confidence=0.5,
            ),
            "aLEFTb",
        )
        self.assertEqual(
            apply_contextual_mapping(
                "axc",
                base_mapping,
                context_counts,
                min_count=2,
                min_confidence=0.5,
            ),
            "aRIGHTc",
        )

    def test_contextual_mapping_falls_back_when_evidence_is_weak(self):
        base_mapping = {"a": "a", "x": "X", "b": "b"}
        context_counts = {
            ("tri", "a", "x", "b"): Counter({"LEFT": 1}),
        }

        self.assertEqual(
            apply_contextual_mapping(
                "axb",
                base_mapping,
                context_counts,
                min_count=2,
                min_confidence=0.5,
            ),
            "aXb",
        )

    def test_contextual_tuning_prefers_stronger_evidence_when_cer_ties(self):
        metrics = {"micro_cer": 0.1, "exact_match": 0.2}

        self.assertLess(tuning_key(metrics, min_count=3), tuning_key(metrics, min_count=1))

    def test_split_syllable_units_preserves_tibetan_delimiters(self):
        units = split_syllable_units("ཧན་ཧོན་དྲན་པ།")

        self.assertEqual(
            [(unit.core, unit.delimiter) for unit in units],
            [("ཧན", "་"), ("ཧོན", "་"), ("དྲན", "་"), ("པ", "།")],
        )

    def test_likely_tibetan_syllable_accepts_stacked_structures(self):
        self.assertTrue(is_likely_tibetan_syllable("སྙོམས"))
        self.assertTrue(is_likely_tibetan_syllable("གཉིས"))
        self.assertTrue(is_likely_tibetan_syllable("དགྲ"))
        self.assertFalse(is_likely_tibetan_syllable("ིཀ"))

    def test_syllable_vocabulary_correction_inserts_missing_boundaries(self):
        vocab = build_syllable_vocabulary(
            [
                ("src", "པ་མི་གསལ་བར།"),
                ("src", "ཧན་ཧོན་དྲན་པ།"),
            ]
        )

        corrected = apply_syllable_vocabulary_correction(
            "པམིསལར།",
            vocab,
            max_edit_distance=1,
        )

        self.assertEqual(corrected, "པ་མི་གསལ་བར།")

    def test_syllable_vocabulary_correction_preserves_short_existing_units(self):
        vocab = build_syllable_vocabulary(
            [
                ("src", "ངན་ཤུགས་ཧམ་ཁྱེར།"),
                ("src", "ངན་ཤུགས་མོ་ཁྱེར།"),
            ]
        )

        corrected = apply_syllable_vocabulary_correction(
            "ངན་ཤུགས་མྷ་ཁྱེར།",
            vocab,
            max_edit_distance=1,
        )

        self.assertEqual(corrected, "ངན་ཤུགས་མྷ་ཁྱེར།")


if __name__ == "__main__":
    unittest.main()
