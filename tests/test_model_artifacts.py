import json
import unittest
from collections import Counter

from model_artifacts import deserialize_contextual_state, serialize_contextual_state, validate_candidate_alignment
from tibetan_syllable import apply_syllable_vocabulary_correction, build_syllable_vocabulary


class ModelArtifactTests(unittest.TestCase):
    def test_complete_state_round_trip_preserves_rare_outputs_and_correction(self):
        pairs = [("བོད་", "བོད་"), ("ཡིག་", "ཡིག།")]
        vocabulary = build_syllable_vocabulary(pairs)
        contexts = {("tri", "a", "x", "b"): Counter({"common": 8, "second": 3, "third": 2, "rare": 1})}
        state = json.loads(json.dumps(serialize_contextual_state({"x": "X"}, contexts, vocabulary)))
        mapping, restored_contexts, restored_vocabulary = deserialize_contextual_state(state)
        self.assertEqual(mapping, {"x": "X"})
        self.assertEqual(restored_contexts, contexts)
        for text in ["བོད་ཡིག་", "བོ་ཡིག་"]:
            self.assertEqual(
                apply_syllable_vocabulary_correction(text, vocabulary, max_edit_distance=1),
                apply_syllable_vocabulary_correction(text, restored_vocabulary, max_edit_distance=1),
            )

    def test_candidate_alignment_rejects_equal_length_permutation(self):
        primary = [{"source": "a", "reference": "A"}, {"source": "b", "reference": "B"}]
        with self.assertRaises(ValueError):
            validate_candidate_alignment(primary, list(reversed(primary)))
        with self.assertRaises(ValueError):
            validate_candidate_alignment(primary, primary, [("b", "B"), ("a", "A")])
        validate_candidate_alignment(primary, primary, [("a", "A"), ("b", "B")])

    def test_candidate_alignment_rejects_truncation_and_reference_mismatch(self):
        primary = [{"source": "a", "reference": "A"}]
        with self.assertRaises(ValueError):
            validate_candidate_alignment(primary, [])
        with self.assertRaises(ValueError):
            validate_candidate_alignment(primary, [{"source": "a", "reference": "B"}])


if __name__ == "__main__":
    unittest.main()
