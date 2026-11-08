import unittest

from merchant_search_lab import (
    Product,
    RelevanceCase,
    SearchIndex,
    SearchValidationError,
    evaluate_relevance,
)


PRODUCTS = (
    Product("MUG-BLUE", "Blue Stoneware Mug", "Coffee and tea cup", ("kitchen",), 2400, 12),
    Product("MUG-WHITE", "White Travel Mug", "Insulated coffee cup", ("travel",), 3200, 4),
    Product("TEA-BLUE", "Blue Breakfast Tea", "Loose-leaf tea", ("pantry",), 1800, 8),
    Product("APRON", "Canvas Apron", "Kitchen workwear", ("kitchen",), 4500, 2),
)


class RelevanceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = SearchIndex(PRODUCTS)

    def test_report_computes_per_query_and_macro_metrics(self) -> None:
        report = evaluate_relevance(
            self.index,
            (
                RelevanceCase("blue mug", ("MUG-BLUE",)),
                RelevanceCase("kitchen", ("MUG-BLUE", "APRON")),
                RelevanceCase("unknown", ("TEA-BLUE",)),
            ),
            k=2,
        )

        self.assertEqual(report.k, 2)
        self.assertEqual(report.query_count, 3)
        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-BLUE", "MUG-WHITE"))
        self.assertEqual(report.queries[0].precision_at_k, 0.5)
        self.assertEqual(report.queries[0].recall_at_k, 1.0)
        self.assertEqual(report.queries[0].reciprocal_rank, 1.0)
        self.assertEqual(report.queries[1].relevant_retrieved, 2)
        self.assertEqual(report.queries[1].precision_at_k, 1.0)
        self.assertEqual(report.queries[1].recall_at_k, 1.0)
        self.assertEqual(report.queries[2].retrieved_skus, ())
        self.assertEqual(report.queries[2].reciprocal_rank, 0.0)
        self.assertAlmostEqual(report.macro_precision_at_k, 0.5)
        self.assertAlmostEqual(report.macro_recall_at_k, 2 / 3)
        self.assertAlmostEqual(report.mean_reciprocal_rank, 2 / 3)

    def test_precision_at_k_uses_k_when_fewer_results_are_retrieved(self) -> None:
        report = evaluate_relevance(
            self.index,
            (RelevanceCase("travel", ("MUG-WHITE",)),),
            k=3,
        )
        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-WHITE",))
        self.assertAlmostEqual(report.queries[0].precision_at_k, 1 / 3)

    def test_reciprocal_rank_uses_first_relevant_result(self) -> None:
        report = evaluate_relevance(
            self.index,
            (RelevanceCase("blue", ("TEA-BLUE",)),),
            k=2,
        )
        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-BLUE", "TEA-BLUE"))
        self.assertEqual(report.queries[0].reciprocal_rank, 0.5)

    def test_typo_tolerance_is_explicitly_forwarded(self) -> None:
        case = (RelevanceCase("cofee", ("MUG-BLUE", "MUG-WHITE")),)
        self.assertEqual(evaluate_relevance(self.index, case, k=2).macro_recall_at_k, 0.0)
        self.assertEqual(
            evaluate_relevance(self.index, case, k=2, typo_tolerance=True).macro_recall_at_k,
            1.0,
        )

    def test_generator_input_is_materialized_once(self) -> None:
        cases = (case for case in (RelevanceCase("tea", ("TEA-BLUE",)),))
        self.assertEqual(evaluate_relevance(self.index, cases).query_count, 1)

    def test_invalid_configuration_and_labels_are_rejected(self) -> None:
        invalid_calls = (
            lambda: evaluate_relevance(self.index, ()),
            lambda: evaluate_relevance(self.index, (RelevanceCase("blue", ("MUG-BLUE",)),), k=0),
            lambda: evaluate_relevance(self.index, (RelevanceCase("blue", ("MUG-BLUE",)),), k=True),
            lambda: evaluate_relevance(self.index, (RelevanceCase("", ("MUG-BLUE",)),)),
            lambda: evaluate_relevance(self.index, (RelevanceCase("blue", ()),)),
            lambda: evaluate_relevance(self.index, (RelevanceCase("blue", ("MUG-BLUE", "MUG-BLUE")),)),
            lambda: evaluate_relevance(self.index, (RelevanceCase("blue", ("UNKNOWN",)),)),
            lambda: evaluate_relevance(
                self.index,
                (RelevanceCase("blue", ("MUG-BLUE",)), RelevanceCase("blue", ("TEA-BLUE",))),
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(SearchValidationError):
                call()


if __name__ == "__main__":
    unittest.main()