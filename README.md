# Merchant Search Lab

A dependency-free, deterministic search and ranking lab for a **synthetic** merchant catalog. It demonstrates the correctness boundaries behind a merchant product-search service without requiring platform credentials or customer data.

Independent portfolio project. Not affiliated with Shopify or any merchant platform.

## Capabilities

- immutable in-memory indexing with strict SKU, inventory, and integer-minor-unit price validation;
- Unicode-aware case-folded tokenization;
- deterministic field-aware ranking with exact-title, title, tag, and description weights;
- exact additive per-term score explanations;
- stock, price, and required-tag filters;
- bounded opaque keyset pagination, with cursors bound to the normalized query and filters;
- stable `score DESC, sku ASC` ordering and focused Python 3.11/3.14 tests;
- opt-in one-edit typo tolerance (insert, delete, replace, or adjacent swap) with discounted, explicit score components;
- conservative correction: exact vocabulary terms are never expanded, and each missing query term has at most three lexically selected candidates.
- reproducible synthetic relevance evaluation with per-query and macro precision@k, recall@k, and mean reciprocal rank.
- a strict, bounded JSON benchmark CLI with deterministic machine-readable output and a version-controlled synthetic fixture.
- exact-only versus typo-tolerant ranking comparison with aggregate and per-query metric deltas.
- explicit zero-result and no-relevant-result counts/rates so retrieval coverage failures are not hidden inside averages.

## Run

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
python demo.py
python -m merchant_search_lab.benchmark examples/synthetic_relevance.json --k 3 --typo-tolerance
python -m merchant_search_lab.benchmark examples/synthetic_relevance.json --k 3 --compare-typo-tolerance
```

The demo searches a small synthetic catalog and prints ranked results plus score components as JSON.

## Synthetic relevance evaluation

`evaluate_relevance` accepts explicit `RelevanceCase` labels and reports deterministic top-k metrics. Precision@k always uses `k` as its denominator, recall@k uses every judged-relevant SKU, and reciprocal rank uses the first relevant result within the top k. Inputs are bounded to 1,000 cases and `k <= 100`; duplicate queries, duplicate labels, unknown catalog SKUs, and empty relevance sets are rejected.

The benchmark CLI loads a UTF-8 JSON document containing exact `catalog` and `cases` fields, then prints stable aggregate and query-level JSON suitable for regression diffs. Fixtures are capped at 1 MB and 10,000 products; unknown or extra fields, duplicate JSON keys, non-finite numbers, invalid product values, and invalid labels fail closed. The checked-in example makes a complete invocation reproducible without external services or data.

`--compare-typo-tolerance` evaluates the exact-only baseline and typo-tolerant candidate against the same catalog, labels, `k`, and query order. It reports baseline, candidate, and signed delta values for every metric plus both retrieved-SKU lists per query. Aggregate and per-query diagnostics distinguish a query that retrieved nothing from one that returned results but retrieved no labeled-relevant SKU. The modes are mutually exclusive so a comparison cannot silently use typo tolerance in its baseline.

The labels are hand-authored synthetic fixtures for regression comparisons. They are not clicks, conversions, customer behavior, or evidence of real-world search quality. Coverage rates describe only the bounded fixture, not production traffic. Metric changes should be inspected query by query rather than treated as an automatic release decision.

## Ranking contract

Each unique query term contributes:

| Match | Points |
| --- | ---: |
| normalized full title exactly equals query | 20 |
| each title occurrence | 8 |
| tag presence | 5 |
| each description occurrence | 2 |
| typo-corrected title occurrence | 4 |
| typo-corrected tag presence | 3 |
| typo-corrected description occurrence | 1 |

A product matching any query term is eligible. Filters are applied first. Typo tolerance is off unless `typo_tolerance=True`; corrections require exactly one edit, terms shorter than four characters are not corrected, and an exact catalog-vocabulary term disables correction for that term. Typo explanation terms use `typed->matched` notation. Scores use only catalog text and are intentionally simple enough to audit; no labels, user behavior, personalization, or hidden LLM reasoning are involved.

## Deliberate boundaries

This is a bounded local lab, not a production search engine. The cursor is an encoding, not a signature; an untrusted HTTP boundary should authenticate or sign it. The index is rebuilt in memory, text normalization is deliberately lightweight, and typo candidate generation scans the in-memory vocabulary with fixed query/candidate bounds rather than using a production spelling index. Ranking does not implement stemming, phonetic matching, synonyms, language-specific analysis, access control, merchandising rules, popularity, conversion optimization, or distributed snapshot consistency. Relevance metrics do not estimate online impact, fairness, revenue, or conversion. All examples and labels are synthetic and make no claim about real merchant performance.
