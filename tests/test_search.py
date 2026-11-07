import unittest

from merchant_search_lab import Product, SearchIndex, SearchValidationError


PRODUCTS = (
    Product("MUG-BLUE", "Blue Stoneware Mug", "Hand-finished mug for tea or coffee", ("kitchen", "ceramic"), 2400, 12),
    Product("MUG-WHITE", "White Travel Mug", "Insulated reusable coffee cup", ("kitchen", "travel"), 3200, 0),
    Product("TEA-BLUE", "Blue Breakfast Tea", "Loose-leaf black tea in a reusable tin", ("pantry", "tea"), 1800, 8),
    Product("APRON-01", "Canvas Apron", "Blue workwear for the kitchen", ("kitchen", "textile"), 4500, 4),
    Product("MUG-EXACT", "Blue Mug", "Compact cup", ("gift",), 2100, 3),
)


class SearchIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = SearchIndex(PRODUCTS)

    def test_exact_title_and_field_weights_are_explainable(self) -> None:
        page = self.index.search("blue mug")
        self.assertEqual([hit.sku for hit in page.hits[:2]], ["MUG-EXACT", "MUG-BLUE"])
        exact = page.hits[0]
        self.assertEqual(exact.score, 36)
        self.assertEqual(sum(item.points for item in exact.explanation), exact.score)
        self.assertEqual(
            [(item.field, item.term, item.points) for item in exact.explanation],
            [("title_exact", "blue mug", 20), ("title", "blue", 8), ("title", "mug", 8)],
        )

    def test_ties_use_sku_as_stable_secondary_order(self) -> None:
        index = SearchIndex(
            (
                Product("B", "Blue", "", (), 1, 1),
                Product("A", "Blue", "", (), 1, 1),
            )
        )
        self.assertEqual([hit.sku for hit in index.search("blue").hits], ["A", "B"])

    def test_filters_apply_before_ranking(self) -> None:
        page = self.index.search(
            "blue kitchen",
            in_stock_only=True,
            min_price_minor=2000,
            max_price_minor=3000,
            required_tags=("kitchen",),
        )
        self.assertEqual([hit.sku for hit in page.hits], ["MUG-BLUE"])

    def test_keyset_pages_have_no_duplicates_or_gaps(self) -> None:
        first = self.index.search("blue mug kitchen tea", limit=2)
        self.assertIsNotNone(first.next_cursor)
        second = self.index.search("blue mug kitchen tea", limit=2, cursor=first.next_cursor)
        third = self.index.search("blue mug kitchen tea", limit=2, cursor=second.next_cursor)
        combined = first.hits + second.hits + third.hits
        unpaged = self.index.search("blue mug kitchen tea", limit=100)
        self.assertEqual([hit.sku for hit in combined], [hit.sku for hit in unpaged.hits])
        self.assertEqual(len({hit.sku for hit in combined}), len(combined))
        self.assertIsNone(third.next_cursor)

    def test_cursor_is_bound_to_query_and_filters(self) -> None:
        first = self.index.search("blue", limit=1)
        with self.assertRaisesRegex(SearchValidationError, "does not match"):
            self.index.search("mug", limit=1, cursor=first.next_cursor)
        with self.assertRaisesRegex(SearchValidationError, "does not match"):
            self.index.search("blue", limit=1, cursor=first.next_cursor, in_stock_only=True)

    def test_malformed_cursor_is_rejected(self) -> None:
        with self.assertRaisesRegex(SearchValidationError, "cursor is invalid"):
            self.index.search("blue", cursor="not!base64")

    def test_unicode_tokenization_is_casefolded(self) -> None:
        index = SearchIndex((Product("CAFE", "CAFÉ Bowl", "", (), 100, 1),))
        self.assertEqual(index.search("café").hits[0].sku, "CAFE")

    def test_no_match_returns_an_empty_page(self) -> None:
        self.assertEqual(self.index.search("unfindable").hits, ())
        self.assertIsNone(self.index.search("unfindable").next_cursor)

    def test_search_validation_rejects_unsafe_bounds(self) -> None:
        invalid_calls = (
            lambda: self.index.search(" "),
            lambda: self.index.search("blue", limit=0),
            lambda: self.index.search("blue", limit=True),
            lambda: self.index.search("blue", min_price_minor=-1),
            lambda: self.index.search("blue", min_price_minor=10, max_price_minor=9),
            lambda: self.index.search("blue", required_tags=("",)),
            lambda: self.index.search("x" * 513),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(SearchValidationError):
                call()

    def test_catalog_validation_rejects_duplicates_and_invalid_money(self) -> None:
        with self.assertRaisesRegex(SearchValidationError, "duplicate sku"):
            SearchIndex((PRODUCTS[0], PRODUCTS[0]))
        with self.assertRaisesRegex(SearchValidationError, "price_minor"):
            SearchIndex((Product("X", "X", "", (), -1, 0),))
        with self.assertRaisesRegex(SearchValidationError, "inventory"):
            SearchIndex((Product("X", "X", "", (), 1, True),))


if __name__ == "__main__":
    unittest.main()
