from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .search import SearchIndex, SearchValidationError

_MAX_EVALUATION_CASES = 1_000
_MAX_EVALUATION_K = 100


@dataclass(frozen=True)
class RelevanceCase:
    """A synthetic query and its explicitly judged relevant product SKUs."""

    query: str
    relevant_skus: tuple[str, ...]


@dataclass(frozen=True)
class QueryMetrics:
    query: str
    relevant_count: int
    retrieved_skus: tuple[str, ...]
    relevant_retrieved: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationReport:
    k: int
    query_count: int
    macro_precision_at_k: float
    macro_recall_at_k: float
    mean_reciprocal_rank: float
    zero_result_count: int
    zero_result_rate: float
    no_relevant_result_count: int
    no_relevant_result_rate: float
    queries: tuple[QueryMetrics, ...]


def evaluate_relevance(
    index: SearchIndex,
    cases: Iterable[RelevanceCase],
    *,
    k: int = 10,
    typo_tolerance: bool = False,
) -> EvaluationReport:
    """Evaluate deterministic top-k ranking against explicit synthetic labels.

    Precision uses ``k`` as its denominator even when fewer than ``k`` results
    are retrieved. Recall uses every relevant SKU in the case. Reciprocal rank
    is zero when no relevant product appears in the top-k results.
    """
    if not isinstance(index, SearchIndex):
        raise SearchValidationError("index must be a SearchIndex")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= _MAX_EVALUATION_K:
        raise SearchValidationError(f"k must be an integer from 1 to {_MAX_EVALUATION_K}")
    if not isinstance(typo_tolerance, bool):
        raise SearchValidationError("typo_tolerance must be a boolean")

    materialized = tuple(cases)
    if not materialized:
        raise SearchValidationError("cases must contain at least one relevance case")
    if len(materialized) > _MAX_EVALUATION_CASES:
        raise SearchValidationError(f"cases must contain at most {_MAX_EVALUATION_CASES} relevance cases")

    query_metrics: list[QueryMetrics] = []
    seen_queries: set[str] = set()
    catalog_skus = set(index.skus)
    for case in materialized:
        if not isinstance(case, RelevanceCase):
            raise SearchValidationError("cases must contain RelevanceCase values")
        if not isinstance(case.query, str) or not case.query.strip():
            raise SearchValidationError("case query must be a non-empty string")
        if case.query in seen_queries:
            raise SearchValidationError(f"duplicate case query: {case.query}")
        seen_queries.add(case.query)
        if (
            not isinstance(case.relevant_skus, tuple)
            or not case.relevant_skus
            or any(not isinstance(sku, str) or not sku.strip() for sku in case.relevant_skus)
        ):
            raise SearchValidationError("relevant_skus must be a non-empty tuple of non-empty strings")
        if len(set(case.relevant_skus)) != len(case.relevant_skus):
            raise SearchValidationError("relevant_skus must not contain duplicates")
        unknown_skus = sorted(set(case.relevant_skus) - catalog_skus)
        if unknown_skus:
            raise SearchValidationError(f"relevant_skus contain unknown catalog SKUs: {', '.join(unknown_skus)}")

        retrieved = tuple(
            hit.sku
            for hit in index.search(case.query, limit=k, typo_tolerance=typo_tolerance).hits
        )
        relevant = set(case.relevant_skus)
        relevant_retrieved = sum(sku in relevant for sku in retrieved)
        first_relevant_rank = next(
            (rank for rank, sku in enumerate(retrieved, start=1) if sku in relevant),
            None,
        )
        query_metrics.append(
            QueryMetrics(
                query=case.query,
                relevant_count=len(relevant),
                retrieved_skus=retrieved,
                relevant_retrieved=relevant_retrieved,
                precision_at_k=relevant_retrieved / k,
                recall_at_k=relevant_retrieved / len(relevant),
                reciprocal_rank=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
            )
        )

    count = len(query_metrics)
    zero_result_count = sum(not item.retrieved_skus for item in query_metrics)
    no_relevant_result_count = sum(item.relevant_retrieved == 0 for item in query_metrics)
    return EvaluationReport(
        k=k,
        query_count=count,
        macro_precision_at_k=sum(item.precision_at_k for item in query_metrics) / count,
        macro_recall_at_k=sum(item.recall_at_k for item in query_metrics) / count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in query_metrics) / count,
        zero_result_count=zero_result_count,
        zero_result_rate=zero_result_count / count,
        no_relevant_result_count=no_relevant_result_count,
        no_relevant_result_rate=no_relevant_result_count / count,
        queries=tuple(query_metrics),
    )
