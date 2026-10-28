import unittest

from ablate_abbrev_restore import (
    add_comparison_fields,
    build_ablation_report,
    detection_metrics_from_predictions,
)


class AblationExperimentTests(unittest.TestCase):
    def test_detection_metrics_from_predictions_counts_confusion_matrix(self):
        examples = [
            {"alignment": [{"label": 1}, {"label": 0}, {"label": 1}]},
            {"alignment": [{"label": 0}]},
        ]

        metrics = detection_metrics_from_predictions(examples, [[1, 1, 0], [0]])

        self.assertEqual(metrics["tokens"], 4)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_add_comparison_fields_uses_full_and_baseline_rows(self):
        rows = [
            {
                "variant": "identity_source",
                "split": "test",
                "restore_micro_cer": 0.4,
                "restore_exact_match": 0.1,
            },
            {
                "variant": "full_two_stage",
                "split": "test",
                "restore_micro_cer": 0.25,
                "restore_exact_match": 0.2,
            },
            {
                "variant": "no_syllable_correction",
                "split": "test",
                "restore_micro_cer": 0.3,
                "restore_exact_match": 0.15,
            },
        ]

        enriched = add_comparison_fields(rows)
        by_variant = {row["variant"]: row for row in enriched}

        self.assertAlmostEqual(by_variant["full_two_stage"]["cer_delta_vs_identity"], -0.15)
        self.assertAlmostEqual(
            by_variant["full_two_stage"]["cer_relative_reduction_vs_identity"], 0.375
        )
        self.assertAlmostEqual(by_variant["no_syllable_correction"]["cer_delta_vs_full"], 0.05)
        self.assertAlmostEqual(by_variant["no_syllable_correction"]["exact_delta_vs_full"], -0.05)

    def test_build_ablation_report_contains_key_tables(self):
        rows = [
            {
                "variant": "identity_source",
                "split": "test",
                "description": "copy source",
                "restore_exact_match": 0.1,
                "restore_micro_cer": 0.4,
                "restore_micro_character_accuracy": 0.6,
                "detection_f1": None,
                "cer_delta_vs_full": 0.15,
                "cer_relative_reduction_vs_identity": 0.0,
            },
            {
                "variant": "full_two_stage",
                "split": "test",
                "description": "all components",
                "restore_exact_match": 0.2,
                "restore_micro_cer": 0.25,
                "restore_micro_character_accuracy": 0.75,
                "detection_f1": 0.8,
                "cer_delta_vs_full": 0.0,
                "cer_relative_reduction_vs_identity": 0.375,
            },
        ]

        report = build_ablation_report(rows, threshold_rows=[], metadata={"test_size": 2})

        self.assertIn("# 藏文缩略词识别与还原消融实验报告", report)
        self.assertIn("identity_source", report)
        self.assertIn("full_two_stage", report)
        self.assertIn("CER", report)


if __name__ == "__main__":
    unittest.main()
