from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_QUERY_BYTES = 512
_MAX_QUERY_TERMS = 32
_MAX_LIMIT = 100
_MAX_TYPO_CANDIDATES = 3
_MIN_TYPO_TERM_LENGTH = 4


class SearchValidationError(ValueError):
    """Raised when catalog or search input violates the public contract."""


@dataclass(frozen=True)
class Product:
    sku: str
    title: str
    description: str
    tags: tuple[str, ...]
    price_minor: int
    inventory: int


@dataclass(frozen=True)
class ScoreComponent:
    field: str
    term: str
    points: int


@dataclass(frozen=True)
class SearchHit:
    sku: str
    title: str
    price_minor: int
    inventory: int
    score: int
    explanation: tuple[ScoreComponent, ...]


@dataclass(frozen=True)
class SearchPage:
    hits: tuple[SearchHit, ...]
    next_cursor: str | None


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.casefold()))


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SearchValidationError(f"{name} must be a non-empty string")
    return value


def _is_single_edit(left: str, right: str) -> bool:
    """Return whether terms differ by one insert, delete, replace, or swap."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
        if len(differences) == 1:
            return True
        return (
            len(differences) == 2
            and differences[1] == differences[0] + 1
            and left[differences[0]] == right[differences[1]]
            and left[differences[1]] == right[differences[0]]
        )
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    skipped = False
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
        elif skipped:
            return False
        else:
            skipped = True
            long_index += 1
    return True


class SearchIndex:
    """Immutable in-memory index for a bounded synthetic merchant catalog."""

    def __init__(self, products: Iterable[Product]) -> None:
        indexed: list[tuple[Product, Counter[str], Counter[str], frozenset[str]]] = []
        seen: set[str] = set()
        for product in products:
            if not isinstance(product, Product):
                raise SearchValidationError("products must contain Product values")
            sku = _require_text("sku", product.sku)
            _require_text("title", product.title)
            if not isinstance(product.description, str):
                raise SearchValidationError("description must be a string")
            if isinstance(product.price_minor, bool) or not isinstance(product.price_minor, int) or product.price_minor < 0:
                raise SearchValidationError("price_minor must be a non-negative integer")
            if isinstance(product.inventory, bool) or not isinstance(product.inventory, int) or product.inventory < 0:
                raise SearchValidationError("inventory must be a non-negative integer")
            if sku in seen:
                raise SearchValidationError(f"duplicate sku: {sku}")
            if not isinstance(product.tags, tuple) or any(not isinstance(tag, str) or not tag.strip() for tag in product.tags):
                raise SearchValidationError("tags must be a tuple of non-empty strings")
            seen.add(sku)
            title_terms = Counter(_tokens(product.title))
            description_terms = Counter(_tokens(product.description))
            tag_terms = frozenset(term for tag in product.tags for term in _tokens(tag))
            indexed.append((product, title_terms, description_terms, tag_terms))
        self._products = tuple(indexed)
        self._vocabulary = tuple(
            sorted(
                {
                    term
                    for _, title_terms, description_terms, tag_terms in indexed
                    for term in (*title_terms, *description_terms, *tag_terms)
                }
            )
        )

    @property
    def skus(self) -> tuple[str, ...]:
        """Return catalog SKUs in deterministic index-insertion order."""
        return tuple(product.sku for product, _, _, _ in self._products)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        cursor: str | None = None,
        in_stock_only: bool = False,
        min_price_minor: int | None = None,
        max_price_minor: int | None = None,
        required_tags: tuple[str, ...] = (),
        typo_tolerance: bool = False,
    ) -> SearchPage:
        _require_text("query", query)
        if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
            raise SearchValidationError(f"query must be at most {_MAX_QUERY_BYTES} UTF-8 bytes")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
            raise SearchValidationError(f"limit must be an integer from 1 to {_MAX_LIMIT}")
        if not isinstance(in_stock_only, bool):
            raise SearchValidationError("in_stock_only must be a boolean")
        if not isinstance(typo_tolerance, bool):
            raise SearchValidationError("typo_tolerance must be a boolean")
        for name, value in (("min_price_minor", min_price_minor), ("max_price_minor", max_price_minor)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise SearchValidationError(f"{name} must be a non-negative integer or None")
        if min_price_minor is not None and max_price_minor is not None and min_price_minor > max_price_minor:
            raise SearchValidationError("min_price_minor cannot exceed max_price_minor")
        if not isinstance(required_tags, tuple) or any(not isinstance(tag, str) or not tag.strip() for tag in required_tags):
            raise SearchValidationError("required_tags must be a tuple of non-empty strings")

        query_terms = tuple(dict.fromkeys(_tokens(query)))
        if not query_terms:
            raise SearchValidationError("query must contain at least one letter or number")
        if len(query_terms) > _MAX_QUERY_TERMS:
            raise SearchValidationError(f"query must contain at most {_MAX_QUERY_TERMS} unique terms")
        required_tag_terms = tuple(sorted(set(term for tag in required_tags for term in _tokens(tag))))
        fingerprint = self._fingerprint(
            query_terms,
            in_stock_only,
            min_price_minor,
            max_price_minor,
            required_tag_terms,
            typo_tolerance,
        )
        after = self._decode_cursor(cursor, fingerprint) if cursor is not None else None

        typo_candidates: dict[str, tuple[str, ...]] = {}
        if typo_tolerance:
            vocabulary = set(self._vocabulary)
            for term in query_terms:
                if term in vocabulary or not _MIN_TYPO_TERM_LENGTH <= len(term) <= 64:
                    continue
                typo_candidates[term] = tuple(
                    candidate
                    for candidate in self._vocabulary
                    if _MIN_TYPO_TERM_LENGTH <= len(candidate) <= 64 and _is_single_edit(term, candidate)
                )[:_MAX_TYPO_CANDIDATES]

        hits: list[SearchHit] = []
        normalized_query = " ".join(query_terms)
        for product, title_terms, description_terms, tag_terms in self._products:
            if in_stock_only and product.inventory == 0:
                continue
            if min_price_minor is not None and product.price_minor < min_price_minor:
                continue
            if max_price_minor is not None and product.price_minor > max_price_minor:
                continue
            if any(term not in tag_terms for term in required_tag_terms):
                continue

            components: list[ScoreComponent] = []
            if " ".join(_tokens(product.title)) == normalized_query:
                components.append(ScoreComponent("title_exact", normalized_query, 20))
            for term in query_terms:
                title_points = title_terms[term] * 8
                tag_points = 5 if term in tag_terms else 0
                description_points = description_terms[term] * 2
                if title_points:
                    components.append(ScoreComponent("title", term, title_points))
                if tag_points:
                    components.append(ScoreComponent("tag", term, tag_points))
                if description_points:
                    components.append(ScoreComponent("description", term, description_points))
                for candidate in typo_candidates.get(term, ()):
                    display_term = f"{term}->{candidate}"
                    typo_title_points = title_terms[candidate] * 4
                    typo_tag_points = 3 if candidate in tag_terms else 0
                    typo_description_points = description_terms[candidate]
                    if typo_title_points:
                        components.append(ScoreComponent("typo_title", display_term, typo_title_points))
                    if typo_tag_points:
                        components.append(ScoreComponent("typo_tag", display_term, typo_tag_points))
                    if typo_description_points:
                        components.append(
                            ScoreComponent("typo_description", display_term, typo_description_points)
                        )
            score = sum(component.points for component in components)
            if score:
                hits.append(
                    SearchHit(
                        sku=product.sku,
                        title=product.title,
                        price_minor=product.price_minor,
                        inventory=product.inventory,
                        score=score,
                        explanation=tuple(components),
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.sku))
        if after is not None:
            hits = [hit for hit in hits if (-hit.score, hit.sku) > (-after[0], after[1])]
        page_hits = tuple(hits[:limit])
        next_cursor = None
        if len(hits) > limit:
            last = page_hits[-1]
            next_cursor = self._encode_cursor(fingerprint, last.score, last.sku)
        return SearchPage(page_hits, next_cursor)

    @staticmethod
    def _fingerprint(
        query_terms: tuple[str, ...],
        in_stock_only: bool,
        min_price_minor: int | None,
        max_price_minor: int | None,
        required_tag_terms: tuple[str, ...],
        typo_tolerance: bool,
    ) -> str:
        request = json.dumps(
            [query_terms, in_stock_only, min_price_minor, max_price_minor, required_tag_terms, typo_tolerance],
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(request.encode("ascii")).hexdigest()[:16]

    @staticmethod
    def _encode_cursor(fingerprint: str, score: int, sku: str) -> str:
        payload = json.dumps([1, fingerprint, score, sku], separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")

    @staticmethod
    def _decode_cursor(cursor: str, fingerprint: str) -> tuple[int, str]:
        if not isinstance(cursor, str) or not cursor or len(cursor) > 1024:
            raise SearchValidationError("cursor is invalid")
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(base64.b64decode(cursor + padding, altchars=b"-_", validate=True))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise SearchValidationError("cursor is invalid") from exc
        if (
            not isinstance(payload, list)
            or len(payload) != 4
            or payload[0] != 1
            or payload[1] != fingerprint
            or isinstance(payload[2], bool)
            or not isinstance(payload[2], int)
            or payload[2] <= 0
            or not isinstance(payload[3], str)
            or not payload[3]
        ):
            raise SearchValidationError("cursor does not match this search")
        return payload[2], payload[3]
