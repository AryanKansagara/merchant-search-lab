import json

from merchant_search_lab import Product, SearchIndex


CATALOG = (
    Product("MUG-BLUE", "Blue Stoneware Mug", "Hand-finished mug for tea or coffee", ("kitchen", "ceramic"), 2400, 12),
    Product("MUG-WHITE", "White Travel Mug", "Insulated reusable coffee cup", ("kitchen", "travel"), 3200, 0),
    Product("TEA-BLUE", "Blue Breakfast Tea", "Loose-leaf black tea in a reusable tin", ("pantry", "tea"), 1800, 8),
    Product("APRON-01", "Canvas Apron", "Blue workwear for the kitchen", ("kitchen", "textile"), 4500, 4),
    Product("MUG-EXACT", "Blue Mug", "Compact cup", ("gift",), 2100, 3),
)


def main() -> None:
    page = SearchIndex(CATALOG).search("blue mug", in_stock_only=True, limit=3)
    print(
        json.dumps(
            {
                "synthetic_only": True,
                "results": [
                    {
                        "sku": hit.sku,
                        "title": hit.title,
                        "score": hit.score,
                        "explanation": [component.__dict__ for component in hit.explanation],
                    }
                    for hit in page.hits
                ],
                "next_cursor": page.next_cursor,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
