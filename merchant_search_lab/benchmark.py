from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO, cast

from .evaluation import EvaluationReport, RelevanceCase, evaluate_relevance
from .search import Product, SearchIndex, SearchValidationError

_MAX_FIXTURE_BYTES = 1_000_000
_MAX_CATALOG_PRODUCTS = 10_000
_PRODUCT_KEYS = frozenset({"sku", "title", "description", "tags", "price_minor", "inventory"})
_CASE_KEYS = frozenset({"query", "relevant_skus"})


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SearchValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise SearchValidationError(f"non-finite JSON number is not allowed: {value}")


def _require_exact_keys(value: object, expected: frozenset[str], context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SearchValidationError(f"{context} must be a JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise SearchValidationError(f"{context} fields are invalid ({'; '.join(details)})")
    return value


def load_fixture(path: str | Path) -> tuple[tuple[Product, ...], tuple[RelevanceCase, ...]]:
    """Load a bounded, strict synthetic catalog and relevance fixture."""
    fixture_path = Path(path)
    if fixture_path.stat().st_size > _MAX_FIXTURE_BYTES:
        raise SearchValidationError(f"fixture must be at most {_MAX_FIXTURE_BYTES} bytes")
    document = json.loads(
        fixture_path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    root = _require_exact_keys(document, frozenset({"catalog", "cases"}), "fixture")

    catalog_values = root["catalog"]
    if not isinstance(catalog_values, list) or not catalog_values:
        raise SearchValidationError("catalog must be a non-empty JSON array")
    if len(catalog_values) > _MAX_CATALOG_PRODUCTS:
        raise SearchValidationError(f"catalog must contain at most {_MAX_CATALOG_PRODUCTS} products")
    products: list[Product] = []
    for index, raw_product in enumerate(catalog_values):
        product = _require_exact_keys(raw_product, _PRODUCT_KEYS, f"catalog[{index}]")
        tags = product["tags"]
        if not isinstance(tags, list):
            raise SearchValidationError(f"catalog[{index}].tags must be a JSON array")
        products.append(
            Product(
                sku=cast(str, product["sku"]),
                title=cast(str, product["title"]),
                description=cast(str, product["description"]),
                tags=tuple(tags),
                price_minor=cast(int, product["price_minor"]),
                inventory=cast(int, product["inventory"]),
            )
        )

    case_values = root["cases"]
    if not isinstance(case_values, list) or not case_values:
        raise SearchValidationError("cases must be a non-empty JSON array")
    cases: list[RelevanceCase] = []
    for index, raw_case in enumerate(case_values):
        case = _require_exact_keys(raw_case, _CASE_KEYS, f"cases[{index}]")
        relevant_skus = case["relevant_skus"]
        if not isinstance(relevant_skus, list):
            raise SearchValidationError(f"cases[{index}].relevant_skus must be a JSON array")
        cases.append(
            RelevanceCase(query=cast(str, case["query"]), relevant_skus=tuple(relevant_skus))
        )

    # Constructing the index here and evaluating in run_benchmark provide the
    # canonical field, duplicate-SKU, label, query, and case-count contracts.
    SearchIndex(products)
    return tuple(products), tuple(cases)


def report_document(report: EvaluationReport, *, typo_tolerance: bool) -> dict[str, object]:
    """Convert an evaluation report into a stable JSON-compatible document."""
    return {
        "synthetic_only": True,
        "configuration": {"k": report.k, "typo_tolerance": typo_tolerance},
        "aggregate": {
            "query_count": report.query_count,
            "macro_precision_at_k": report.macro_precision_at_k,
            "macro_recall_at_k": report.macro_recall_at_k,
            "mean_reciprocal_rank": report.mean_reciprocal_rank,
        },
        "queries": [
            {
                "query": item.query,
                "relevant_count": item.relevant_count,
                "retrieved_skus": list(item.retrieved_skus),
                "relevant_retrieved": item.relevant_retrieved,
                "precision_at_k": item.precision_at_k,
                "recall_at_k": item.recall_at_k,
                "reciprocal_rank": item.reciprocal_rank,
            }
            for item in report.queries
        ],
    }


def run_benchmark(path: str | Path, *, k: int, typo_tolerance: bool) -> dict[str, object]:
    products, cases = load_fixture(path)
    report = evaluate_relevance(
        SearchIndex(products),
        cases,
        k=k,
        typo_tolerance=typo_tolerance,
    )
    return report_document(report, typo_tolerance=typo_tolerance)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic search against a strict synthetic relevance fixture."
    )
    parser.add_argument("fixture", help="path to the synthetic JSON fixture")
    parser.add_argument("--k", type=int, default=10, help="top-k cutoff (default: 10; maximum: 100)")
    parser.add_argument("--typo-tolerance", action="store_true", help="enable bounded one-edit typo matching")
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    args = _parser().parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        document = run_benchmark(args.fixture, k=args.k, typo_tolerance=args.typo_tolerance)
    except (OSError, UnicodeError, json.JSONDecodeError, SearchValidationError) as exc:
        print(f"error: {exc}", file=errors)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
