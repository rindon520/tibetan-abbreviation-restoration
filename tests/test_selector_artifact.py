import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from infer_selector import predict_candidates
from train_hybrid_selector import CharacterLanguageModel, selector_features


class SelectorArtifactTests(unittest.TestCase):
    def test_json_coefficients_reproduce_pipeline_without_reference_features(self):
        language_model = CharacterLanguageModel([f"abc{i}" for i in range(20)])
        vocabulary = ["abc1", "abc2", "abc3"]
        primary = [{"source": f"abc{i}", "prediction": f"abc{i}", "reference": "unused"} for i in range(20)]
        auxiliary = [{"source": row["source"], "prediction": f"xyz{i}", "reference": "unused"} for i, row in enumerate(primary)]
        features = np.asarray([selector_features(first, second, language_model, vocabulary) for first, second in zip(primary, auxiliary)])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, random_state=42)).fit(features, np.asarray([i % 2 for i in range(20)]))
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        state = {
            "format_version": 1, "threshold": 0.5,
            "feature_count": features.shape[1], "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(), "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]), "vocabulary": vocabulary,
            "language_model": {
                "max_order": language_model.max_order, "alpha": language_model.alpha,
                "alphabet_size": language_model.alphabet_size,
                "counts": {str(order): dict(counts) for order, counts in language_model.counts.items()},
                "context_counts": {str(order): dict(counts) for order, counts in language_model.context_counts.items()},
            },
        }
        predictions = predict_candidates(state, primary, auxiliary)
        np.testing.assert_allclose(
            [row["auxiliary_probability"] for row in predictions],
            model.predict_proba(features)[:, 1], rtol=1e-12, atol=1e-12,
        )
        for rows in (primary, auxiliary):
            for row in rows:
                row.pop("reference")
        self.assertEqual(predictions, predict_candidates(state, primary, auxiliary))


if __name__ == "__main__":
    unittest.main()
