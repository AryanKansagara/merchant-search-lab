import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, cast

from merchant_search_lab import SearchValidationError
from merchant_search_lab.benchmark import load_fixture, main, run_benchmark, run_comparison


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
        {
            "query": "blue mug",
            "category": "exact_product_intent",
            "relevant_skus": ["MUG-BLUE"],
        },
        {
            "query": "cofee",
            "category": "misspelling",
            "relevant_skus": ["MUG-BLUE", "MUG-WHITE"],
        },
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
        categories = cast(list[dict[str, Any]], document["categories"])
        queries = cast(list[dict[str, Any]], document["queries"])
        self.assertIsInstance(configuration, dict)
        self.assertIsInstance(aggregate, dict)
        self.assertIsInstance(queries, list)
        self.assertEqual(configuration, {"k": 2, "typo_tolerance": True})
        self.assertEqual(aggregate["query_count"], 2)
        self.assertEqual(aggregate["zero_result_count"], 0)
        self.assertEqual(aggregate["no_relevant_result_count"], 0)
        self.assertEqual(
            [item["category"] for item in categories],
            ["exact_product_intent", "misspelling"],
        )
        self.assertEqual(categories[0]["query_count"], 1)
        self.assertEqual(queries[0]["category"], "exact_product_intent")
        self.assertEqual(queries[0]["retrieved_skus"], ["MUG-BLUE", "MUG-WHITE"])
        self.assertFalse(queries[0]["zero_results"])
        self.assertFalse(queries[0]["no_relevant_results"])
        self.assertEqual(queries[1]["recall_at_k"], 1.0)

    def test_cli_writes_json_and_returns_success(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()

        status = main([str(self.fixture_path), "--k", "2"], stdout=output, stderr=errors)

        self.assertEqual(status, 0)
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(json.loads(output.getvalue())["configuration"]["k"], 2)

    def test_comparison_reports_aggregate_and_query_deltas(self) -> None:
        document = run_comparison(self.fixture_path, k=2)

        self.assertEqual(document["mode"], "typo_tolerance_comparison")
        configuration = cast(dict[str, Any], document["configuration"])
        aggregate = cast(dict[str, Any], document["aggregate"])
        categories = cast(list[dict[str, Any]], document["categories"])
        queries = cast(list[dict[str, Any]], document["queries"])
        self.assertEqual(configuration["baseline"], {"typo_tolerance": False})
        self.assertEqual(configuration["candidate"], {"typo_tolerance": True})
        self.assertEqual(
            aggregate["macro_recall_at_k"],
            {"baseline": 0.5, "candidate": 1.0, "delta": 0.5},
        )
        self.assertEqual(
            aggregate["zero_result_count"],
            {"baseline": 1, "candidate": 0, "delta": -1},
        )
        self.assertEqual(
            aggregate["no_relevant_result_rate"],
            {"baseline": 0.5, "candidate": 0.0, "delta": -0.5},
        )
        self.assertEqual(queries[0]["baseline_retrieved_skus"], queries[0]["candidate_retrieved_skus"])
        self.assertEqual(queries[0]["recall_at_k"]["delta"], 0.0)
        self.assertEqual(queries[1]["baseline_retrieved_skus"], [])
        self.assertEqual(queries[1]["candidate_retrieved_skus"], ["MUG-BLUE", "MUG-WHITE"])
        self.assertEqual(queries[1]["zero_results"], {"baseline": True, "candidate": False})
        self.assertEqual(
            queries[1]["no_relevant_results"], {"baseline": True, "candidate": False}
        )
        self.assertEqual(queries[1]["recall_at_k"]["delta"], 1.0)
        self.assertEqual(categories[0]["macro_recall_at_k"]["delta"], 0.0)
        self.assertEqual(categories[1]["macro_recall_at_k"]["delta"], 1.0)

    def test_cli_comparison_is_deterministic(self) -> None:
        first = io.StringIO()
        second = io.StringIO()

        self.assertEqual(
            main([str(self.fixture_path), "--k", "2", "--compare-typo-tolerance"], stdout=first),
            0,
        )
        self.assertEqual(
            main([str(self.fixture_path), "--k", "2", "--compare-typo-tolerance"], stdout=second),
            0,
        )
        self.assertEqual(first.getvalue(), second.getvalue())

    def test_comparison_and_single_candidate_modes_are_exclusive(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            main([str(self.fixture_path), "--typo-tolerance", "--compare-typo-tolerance"])
        self.assertIn("not allowed with argument", errors.getvalue())

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
        invalid["cases"] = [
            {"query": "mug", "category": "exact", "relevant_skus": ["UNKNOWN"]}
        ]
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

    def test_categories_are_required_and_canonically_validated(self) -> None:
        invalid = json.loads(json.dumps(FIXTURE))
        del invalid["cases"][0]["category"]
        self.fixture_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(SearchValidationError, "missing category"):
            load_fixture(self.fixture_path)

        invalid = json.loads(json.dumps(FIXTURE))
        invalid["cases"][0]["category"] = " untrimmed "
        self.fixture_path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(SearchValidationError, "trimmed non-empty string"):
            run_benchmark(self.fixture_path, k=2, typo_tolerance=False)


if __name__ == "__main__":
    unittest.main()
