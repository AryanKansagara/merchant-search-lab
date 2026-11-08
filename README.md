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

## Run

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
python demo.py
```

The demo searches a small synthetic catalog and prints ranked results plus score components as JSON.

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

This is a bounded local lab, not a production search engine. The cursor is an encoding, not a signature; an untrusted HTTP boundary should authenticate or sign it. The index is rebuilt in memory, text normalization is deliberately lightweight, and typo candidate generation scans the in-memory vocabulary with fixed query/candidate bounds rather than using a production spelling index. Ranking does not implement stemming, phonetic matching, synonyms, language-specific analysis, access control, merchandising rules, popularity, conversion optimization, or distributed snapshot consistency. All examples are synthetic and make no claim about real merchant performance.
