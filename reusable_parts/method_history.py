"""
method_history.py
------------------
Look up the year a method/technique first appears in the academic literature,
via the Semantic Scholar *bulk* search endpoint (the regular search endpoint
does not support sort=publicationDate, so it can't be used to find the
earliest paper).

This fixes the original notebook bug where `response['data']` raised a
KeyError whenever Semantic Scholar returned an error payload, an empty
result set, or a rate-limit response without a 'data' key.

Usage
-----
    from method_history import get_origin_year, get_origin_years

    year = get_origin_year("Word2vec text analysis")
    years = get_origin_years(["Word2vec text analysis", "Sentiment analysis finance"])
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def get_origin_year(
    search_query: str,
    end_year: Optional[int] = None,
    api_key: Optional[str] = None,
    retries: int = 3,
    backoff: float = 5.0,
) -> Optional[int]:
    """
    Return the publication year of the earliest paper matching search_query,
    or None if no papers are found, the request errors out after retries,
    or the response is malformed.

    api_key defaults to the SEMANTIC_SCHOLAR_API_KEY environment variable.
    Requests work without a key but are subject to a much lower rate limit.
    """
    api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}

    params = {
        "query": search_query,
        "fields": "title,publicationDate,year",
        "sort": "publicationDate:asc",
    }
    if end_year:
        params["year"] = f"-{end_year}"

    for attempt in range(retries):
        try:
            resp = requests.get(BULK_SEARCH_URL, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.RequestException as exc:
            if attempt == retries - 1:
                logger.error("Origin-year lookup failed for '%s': %s", search_query, exc)
                return None
            wait = backoff * (attempt + 1) + random.uniform(0, 2)
            logger.warning("Request failed (%s), retrying in %.1fs…", exc, wait)
            time.sleep(wait)
            continue

        # Defensive: Semantic Scholar can return an error body, a rate-limit
        # message, or an empty result set — none of which have a 'data' key.
        if "data" not in payload:
            logger.warning(
                "No 'data' key in response for '%s' (got keys: %s) — treating as no match.",
                search_query,
                list(payload.keys()),
            )
            return None

        papers = payload.get("data") or []
        if not papers:
            logger.info("No papers found for query: %s", search_query)
            return None

        earliest = papers[0]
        year = earliest.get("year")
        if year is None and earliest.get("publicationDate"):
            try:
                year = int(str(earliest["publicationDate"])[:4])
            except ValueError:
                year = None
        return year

    return None


def get_origin_years(
    search_queries: List[str],
    end_year: Optional[int] = None,
    api_key: Optional[str] = None,
    delay: tuple = (7, 11),
) -> Dict[str, Optional[int]]:
    """
    Batch version of get_origin_year — looks up each query in turn with a
    polite delay between calls to respect API rate limits.

    Returns {search_query: year_or_None}.
    """
    results: Dict[str, Optional[int]] = {}
    for i, query in enumerate(search_queries):
        results[query] = get_origin_year(query, end_year=end_year, api_key=api_key)
        if i < len(search_queries) - 1:
            time.sleep(random.randint(*delay))
    return results
