import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grouped_protocol", ROOT / "scripts/run_grouped_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class GroupedProtocolTests(unittest.TestCase):
    def test_parent_and_duplicate_source_components_never_cross_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "raw"
            data.mkdir()
            sources = ["a  b", "a"] + [f"source{i}" for i in range(38)]
            references = ["A།B།", "A།"] + [f"target{i}།" for i in range(38)]
            (data / "Source.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
            (data / "Restore.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
            args = SimpleNamespace(raw_data_dir=data, output_dir=root / "run", seed=42)
            with contextlib.redirect_stdout(io.StringIO()):
                protocol.prepare(args)
            manifest, records, _ = protocol.load_corpus(args)
            self.assertEqual(records[0]["group_id"], records[1]["group_id"])
            self.assertEqual(records[0]["group_id"], records[2]["group_id"])
            index_to_split = {index: split for split, item in manifest["splits"].items() for index in item["record_indices"]}
            self.assertEqual(index_to_split[0], index_to_split[1])
            self.assertEqual(index_to_split[0], index_to_split[2])
            groups = {split: set(item["group_ids"]) for split, item in manifest["splits"].items()}
            for first in groups:
                for second in groups:
                    if first < second:
                        self.assertFalse(groups[first] & groups[second])
            original = json.loads((args.output_dir / "corpus/split_manifest.json").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                protocol.prepare(args)
            again = json.loads((args.output_dir / "corpus/split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(original, again)
            records_file = args.output_dir / "corpus/records.jsonl"
            records_file.write_text(records_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                protocol.load_corpus(args)


if __name__ == "__main__":
    unittest.main()
