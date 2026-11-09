import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from merchant_search_lab import SearchValidationError
from merchant_search_lab.benchmark import load_fixture, main, run_benchmark


FIXTURE = {
    "catalog": [
        {
            "sku": "MUG-BLUE",
            "title": "Blue Stoneware Mug",
            "description": "Coffee cup",
            "tags": ["kitchen"],
            "price_minor": 2400,
            "inventory": 12,
        },
        {
            "sku": "MUG-WHITE",
            "title": "White Travel Mug",
            "description": "Insulated coffee cup",
            "tags": ["travel"],
            "price_minor": 3200,
            "inventory": 4,
        },
    ],
    "cases": [
        {"query": "blue mug", "relevant_skus": ["MUG-BLUE"]},
        {"query": "cofee", "relevant_skus": ["MUG-BLUE", "MUG-WHITE"]},
    ],
}


class BenchmarkCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture_path = Path(self.temporary_directory.name) / "fixture.json"
        self.fixture_path.write_text(json.dumps(FIXTURE), encoding="utf-8")

    def test_fixture_runs_and_emits_stable_query_details(self) -> None:
        document = run_benchmark(self.fixture_path, k=2, typo_tolerance=True)

        self.assertTrue(document["synthetic_only"])
        configuration = cast(dict[str, object], document["configuration"])
        aggregate = cast(dict[str, object], document["aggregate"])
        queries = cast(list[dict[str, Any]], document["queries"])
        self.assertIsInstance(configuration, dict)
        self.assertIsInstance(aggregate, dict)
        self.assertIsInstance(queries, list)
        self.assertEqual(configuration, {"k": 2, "typo_tolerance": True})
        self.assertEqual(aggregate["query_count"], 2)
        self.assertEqual(queries[0]["retrieved_skus"], ["MUG-BLUE", "MUG-WHITE"])
        self.assertEqual(queries[1]["recall_at_k"], 1.0)

    def test_cli_writes_json_and_returns_success(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()

        status = main([str(self.fixture_path), "--k", "2"], stdout=output, stderr=errors)

        self.assertEqual(status, 0)
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(json.loads(output.getvalue())["configuration"]["k"], 2)

    def test_cli_reports_validation_without_traceback(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()

        status = main([str(self.fixture_path), "--k", "0"], stdout=output, stderr=errors)

        self.assertEqual(status, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("k must be an integer from 1 to 100", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_unknown_labels_are_rejected_by_canonical_evaluation(self) -> None:
        invalid = dict(FIXTURE)
        invalid["cases"] = [{"query": "mug", "relevant_skus": ["UNKNOWN"]}]
        self.fixture_path.write_text(json.dumps(invalid), encoding="utf-8")

        with self.assertRaisesRegex(SearchValidationError, "unknown catalog SKUs"):
            run_benchmark(self.fixture_path, k=2, typo_tolerance=False)

    def test_schema_rejects_extra_fields_and_non_array_tags(self) -> None:
        invalid = json.loads(json.dumps(FIXTURE))
        invalid["catalog"][0]["secret"] = "not allowed"
        self.fixture_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(SearchValidationError, "unexpected secret"):
            load_fixture(self.fixture_path)

        invalid = json.loads(json.dumps(FIXTURE))
        invalid["catalog"][0]["tags"] = "kitchen"
        self.fixture_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(SearchValidationError, "tags must be a JSON array"):
            load_fixture(self.fixture_path)

    def test_duplicate_json_keys_and_oversized_files_are_rejected(self) -> None:
        self.fixture_path.write_text('{"catalog": [], "catalog": [], "cases": []}', encoding="utf-8")
        with self.assertRaisesRegex(SearchValidationError, "duplicate JSON key: catalog"):
            load_fixture(self.fixture_path)

        self.fixture_path.write_bytes(b" " * 1_000_001)
        with self.assertRaisesRegex(SearchValidationError, "at most 1000000 bytes"):
            load_fixture(self.fixture_path)


if __name__ == "__main__":
    unittest.main()
