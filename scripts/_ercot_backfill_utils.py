"""
_ercot_backfill_utils.py — shared pacing/retry helper for the ERCOT
full-year backfill scripts (fetch_ercot_dam_full_year.py,
fetch_ercot_rtm_full_year.py). Not part of the bess package — this is
backfill-script-specific plumbing, not something a normal single-day or
single-range fetch needs.

Added after a real run hit 429 Too Many Requests partway through a
41-request RTM backfill: the requests were fired back-to-back with no
pacing at all, despite already having found evidence (a GitHub discussion
on ERCOT's own api-specs repo, surfaced while debugging the earlier
"socket hang up" issue) that this API rate-limits at roughly 20
requests/minute.
"""

import time
import warnings
from typing import Callable, TypeVar

import requests

T = TypeVar("T")


def fetch_with_retry(
    fetch_fn: Callable[..., T],
    *args,
    max_retries: int = 4,
    pace_seconds: float = 3.5,
    initial_backoff_seconds: float = 15.0,
    **kwargs,
) -> T:
    """
    Call fetch_fn(*args, **kwargs). On success, sleeps `pace_seconds`
    before returning, to keep the *next* call safely under ERCOT's rate
    limit. On a 429 response specifically, waits with exponential backoff
    (starting at `initial_backoff_seconds`) and retries, up to
    `max_retries` times, before giving up and re-raising. Any other error
    (a genuine data problem, a non-429 HTTP error) is not retried — this
    is purely a rate-limit accommodation, not a general-purpose retry.
    """
    wait = initial_backoff_seconds
    for attempt in range(max_retries + 1):
        try:
            result = fetch_fn(*args, **kwargs)
            time.sleep(pace_seconds)
            return result
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and attempt < max_retries:
                warnings.warn(f"rate limited (429) — waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
                wait *= 2
                continue
            raise
