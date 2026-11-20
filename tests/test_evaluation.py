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
        self.assertAlmostEqual(report.macro_ndcg_at_k, 2 / 3)
        self.assertAlmostEqual(report.average_retrieved_count, 4 / 3)
        self.assertEqual(report.catalog_size, 4)
        self.assertEqual(report.unique_retrieved_count, 3)
        self.assertEqual(report.catalog_coverage_rate, 0.75)
        self.assertEqual(report.judged_relevant_sku_count, 3)
        self.assertEqual(report.retrieved_relevant_sku_count, 2)
        self.assertAlmostEqual(report.judged_relevant_coverage_rate, 2 / 3)
        self.assertEqual(report.relevant_judgment_count, 4)
        self.assertEqual(report.retrieved_relevant_judgment_count, 3)
        self.assertEqual(report.micro_recall_at_k, 0.75)
        self.assertEqual(report.underfilled_query_count, 1)
        self.assertAlmostEqual(report.underfilled_query_rate, 1 / 3)
        self.assertEqual(report.zero_result_count, 1)
        self.assertAlmostEqual(report.zero_result_rate, 1 / 3)
        self.assertEqual(report.no_relevant_result_count, 1)
        self.assertAlmostEqual(report.no_relevant_result_rate, 1 / 3)
        self.assertEqual(len(report.categories), 1)
        self.assertEqual(report.categories[0].category, "uncategorized")
        self.assertEqual(report.categories[0].query_count, 3)

    def test_category_metrics_are_sorted_and_computed_per_category(self) -> None:
        report = evaluate_relevance(
            self.index,
            (
                RelevanceCase("unknown", ("TEA-BLUE",), "zero_result"),
                RelevanceCase("blue mug", ("MUG-BLUE",), "exact_intent"),
                RelevanceCase("travel", ("MUG-WHITE",), "exact_intent"),
            ),
            k=2,
        )

        self.assertEqual(
            [item.category for item in report.categories], ["exact_intent", "zero_result"]
        )
        self.assertEqual(report.categories[0].query_count, 2)
        self.assertEqual(report.categories[0].macro_recall_at_k, 1.0)
        self.assertEqual(report.categories[0].average_retrieved_count, 1.5)
        self.assertEqual(report.categories[0].unique_retrieved_count, 2)
        self.assertEqual(report.categories[0].catalog_coverage_rate, 0.5)
        self.assertEqual(report.categories[0].judged_relevant_sku_count, 2)
        self.assertEqual(report.categories[0].retrieved_relevant_sku_count, 2)
        self.assertEqual(report.categories[0].judged_relevant_coverage_rate, 1.0)
        self.assertEqual(report.categories[0].relevant_judgment_count, 2)
        self.assertEqual(report.categories[0].retrieved_relevant_judgment_count, 2)
        self.assertEqual(report.categories[0].micro_recall_at_k, 1.0)
        self.assertEqual(report.categories[0].underfilled_query_count, 1)
        self.assertEqual(report.categories[1].zero_result_rate, 1.0)

    def test_no_relevant_result_is_distinct_from_zero_results(self) -> None:
        report = evaluate_relevance(
            self.index,
            (RelevanceCase("blue mug", ("TEA-BLUE",)),),
            k=1,
        )

        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-BLUE",))
        self.assertEqual(report.zero_result_count, 0)
        self.assertEqual(report.no_relevant_result_count, 1)

    def test_precision_at_k_uses_k_when_fewer_results_are_retrieved(self) -> None:
        report = evaluate_relevance(
            self.index,
            (RelevanceCase("travel", ("MUG-WHITE",)),),
            k=3,
        )
        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-WHITE",))
        self.assertAlmostEqual(report.queries[0].precision_at_k, 1 / 3)

    def test_micro_recall_counts_repeated_sku_judgments_per_query(self) -> None:
        report = evaluate_relevance(
            self.index,
            (
                RelevanceCase("blue mug", ("MUG-BLUE",)),
                RelevanceCase("unknown", ("MUG-BLUE",)),
            ),
            k=1,
        )

        self.assertEqual(report.judged_relevant_sku_count, 1)
        self.assertEqual(report.retrieved_relevant_sku_count, 1)
        self.assertEqual(report.judged_relevant_coverage_rate, 1.0)
        self.assertEqual(report.relevant_judgment_count, 2)
        self.assertEqual(report.retrieved_relevant_judgment_count, 1)
        self.assertEqual(report.micro_recall_at_k, 0.5)

    def test_reciprocal_rank_uses_first_relevant_result(self) -> None:
        report = evaluate_relevance(
            self.index,
            (RelevanceCase("blue", ("TEA-BLUE",)),),
            k=2,
        )
        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-BLUE", "TEA-BLUE"))
        self.assertEqual(report.queries[0].reciprocal_rank, 0.5)
        self.assertAlmostEqual(report.queries[0].ndcg_at_k, 0.6309297535714575)

    def test_ndcg_is_one_for_an_ideal_binary_relevance_ordering(self) -> None:
        report = evaluate_relevance(
            self.index,
            (RelevanceCase("blue", ("MUG-BLUE", "TEA-BLUE")),),
            k=3,
        )

        self.assertEqual(report.queries[0].retrieved_skus, ("MUG-BLUE", "TEA-BLUE"))
        self.assertEqual(report.queries[0].ndcg_at_k, 1.0)
        self.assertEqual(report.categories[0].macro_ndcg_at_k, 1.0)

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
            lambda: evaluate_relevance(
                self.index, (RelevanceCase("blue", ("MUG-BLUE",), " untrimmed "),)
            ),
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
