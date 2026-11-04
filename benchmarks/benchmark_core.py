"""Shared, dependency-free helpers for benchmark timing and statistics."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from typing import Any, Callable


def build_prompt_tokens(tokenizer: Any, requested_length: int) -> list[int]:
    """Build a deterministic prompt containing exactly ``requested_length`` tokens."""
    if requested_length <= 0:
        raise ValueError("Input length must be positive.")

    base_tokens = tokenizer.encode("hello ", add_special_tokens=False)
    if not base_tokens:
        raise ValueError("Tokenizer returned no tokens for the benchmark prompt.")

    repeats = math.ceil(requested_length / len(base_tokens))
    return (base_tokens * repeats)[:requested_length]


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for values in seconds."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure_stream(
    responses: Iterable[Any],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, float | int | None], Any]:
    """Measure TTFT and post-first-token decode performance.

    TTFT covers the interval immediately before stream consumption through the
    first yielded token. Decode throughput excludes that first token and uses
    the wall-clock interval from the first yielded token to the final token.
    """
    start_time = clock()
    token_times: list[float] = []
    first_generation_count: int | None = None
    last_response = None

    for response in responses:
        observed_at = clock()
        if first_generation_count is None:
            first_generation_count = response.generation_tokens
        token_times.append(observed_at)
        last_response = response

    if last_response is None:
        raise RuntimeError("Model generated no tokens.")

    ttft_seconds = token_times[0] - start_time
    total_seconds = token_times[-1] - start_time
    decode_seconds = token_times[-1] - token_times[0]
    decode_tokens = max(
        last_response.generation_tokens - (first_generation_count or 0),
        0,
    )
    inter_token_intervals = [
        current - previous
        for previous, current in zip(token_times, token_times[1:])
    ]

    decode_tps = (
        decode_tokens / decode_seconds
        if decode_tokens > 0 and decode_seconds > 0
        else None
    )
    mean_itl_ms = (
        decode_seconds / decode_tokens * 1000
        if decode_tokens > 0 and decode_seconds > 0
        else None
    )
    p50_itl = percentile(inter_token_intervals, 0.50)
    p95_itl = percentile(inter_token_intervals, 0.95)

    metrics: dict[str, float | int | None] = {
        "ttft_seconds": ttft_seconds,
        "decode_time_seconds": decode_seconds,
        "decode_tokens": decode_tokens,
        "decode_tokens_per_second": decode_tps,
        "mean_inter_token_latency_ms": mean_itl_ms,
        "p50_inter_token_latency_ms": (
            p50_itl * 1000 if p50_itl is not None else None
        ),
        "p95_inter_token_latency_ms": (
            p95_itl * 1000 if p95_itl is not None else None
        ),
        "total_time_seconds": total_seconds,
    }
    return metrics, last_response
