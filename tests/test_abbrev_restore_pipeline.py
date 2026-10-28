import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from abbrev_restore import (
    AbbreviationDetector,
    build_examples,
    build_restoration_table,
    evaluate_detection,
    predict_restoration_rows,
    train_detector,
)
from segment_tibetan import fallback_segment_line, segment_lines
from tibetan_syllable import build_syllable_vocabulary
from train_transducer import learn_char_mapping


class AbbreviationRestorePipelineTests(unittest.TestCase):
    def test_segmenter_runs_in_segmenter_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmenter_dir = root / "segment-ti"
            segmenter_dir.mkdir()

            def fake_run(command, cwd, capture_output, text, encoding, check, timeout):
                self.assertEqual(Path(cwd), segmenter_dir)
                self.assertGreater(timeout, 0)
                self.assertIn(command[0], ("./tiSegmenter", ".\\tiSegmenter"))
                output_path = Path(command[4])
                output_path.write_text("tok1 tok2\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("segment_tibetan.subprocess.run", side_effect=fake_run):
                lines, status = segment_lines(
                    ["tok1tok2"],
                    segmenter_dir=segmenter_dir,
                    prefer_external=True,
                    allow_fallback=False,
                )

        self.assertEqual(lines, ["tok1 tok2"])
        self.assertEqual(status["mode"], "external")
        self.assertEqual(Path(status["cwd"]), segmenter_dir)

    def test_empty_successful_segmenter_output_raises_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                Path(command[4]).write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command, 0, stdout="Segmentor initialize fail", stderr=""
                )

            with patch("segment_tibetan.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "expected 2, got 0"):
                    segment_lines(
                        ["tok1tok2", "tok3tok4"],
                        segmenter_dir=Path(tmp),
                        prefer_external=True,
                        allow_fallback=False,
                    )

    def test_empty_successful_segmenter_output_uses_allowed_fallback(self):
        source_lines = ["tok1tok2", "tok3tok4"]
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                Path(command[4]).write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command, 0, stdout="Segmentor initialize fail", stderr=""
                )

            with patch("segment_tibetan.subprocess.run", side_effect=fake_run):
                lines, status = segment_lines(
                    source_lines,
                    segmenter_dir=Path(tmp),
                    prefer_external=True,
                    allow_fallback=True,
                )

        self.assertEqual(lines, source_lines)
        self.assertEqual(status["mode"], "fallback")
        self.assertEqual(status["lines"], len(source_lines))
        self.assertIn("expected 2, got 0", status["reason"])

    def test_fallback_segment_line_uses_tibetan_syllable_boundaries(self):
        text = "\u0f56\u0f7c\u0f51\u0f0b\u0f61\u0f72\u0f42\u0f0d"

        self.assertEqual(
            fallback_segment_line(text),
            "\u0f56\u0f7c\u0f51\u0f0b \u0f61\u0f72\u0f42\u0f0d",
        )

    def test_build_examples_marks_expanded_token_as_abbreviation(self):
        examples = build_examples(
            pairs=[("a b", "alpha b")],
            source_segmented=["a b"],
            reference_segmented=["alpha b"],
            max_ref_units_per_source=3,
        )

        self.assertEqual(examples[0]["labels"], [1, 0])
        self.assertEqual(examples[0]["alignment"][0]["reference_text"], "alpha")

    def test_detector_and_restorer_recover_seen_expansion(self):
        pairs = [("x y", "expanded y"), ("x z", "expanded z"), ("a y", "a y")]
        examples = build_examples(
            pairs=pairs,
            source_segmented=["x y", "x z", "a y"],
            reference_segmented=["expanded y", "expanded z", "a y"],
            max_ref_units_per_source=3,
        )
        restoration_table = build_restoration_table(examples)
        detector = train_detector(
            examples,
            restoration_table=restoration_table,
            epochs=4,
            seed=7,
        )
        char_mapping, _ = learn_char_mapping(pairs, iterations=2, max_output_length=12)
        vocab = build_syllable_vocabulary(pairs)

        rows = predict_restoration_rows(
            examples,
            detector=detector,
            threshold=0.0,
            restoration_table=restoration_table,
            char_mapping=char_mapping,
            syllable_vocab=vocab,
            max_edit_distance=0,
            max_syllable_chars=12,
        )
        detection = evaluate_detection(examples, detector, threshold=0.0)

        self.assertGreaterEqual(detection["f1"], 0.8)
        self.assertEqual(rows[0]["prediction"], "expandedy")

    def test_terminal_shad_is_applied_after_joining_units(self):
        examples = build_examples(
            pairs=[("x\u0f0b y\u0f0b", "x\u0f0by\u0f0d")],
            source_segmented=["x\u0f0b y\u0f0b"],
            reference_segmented=["x\u0f0b y\u0f0d"],
        )
        detector = AbbreviationDetector(weights={}, restoration_table={})

        rows = predict_restoration_rows(
            examples,
            detector=detector,
            threshold=99.0,
            restoration_table={},
            char_mapping={},
            syllable_vocab=build_syllable_vocabulary([("x", "x\u0f0by\u0f0d")]),
            max_edit_distance=0,
            max_syllable_chars=8,
        )

        self.assertEqual(rows[0]["prediction"], "x\u0f0by\u0f0d")


if __name__ == "__main__":
    unittest.main()
