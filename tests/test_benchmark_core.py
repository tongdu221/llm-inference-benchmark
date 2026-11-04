import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import build_prompt_tokens, measure_stream, percentile


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        self.last_add_special_tokens = add_special_tokens
        return [11, 22]


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class BenchmarkCoreTests(unittest.TestCase):
    def test_prompt_has_exact_requested_length(self):
        tokenizer = FakeTokenizer()
        tokens = build_prompt_tokens(tokenizer, 5)
        self.assertEqual(tokens, [11, 22, 11, 22, 11])
        self.assertFalse(tokenizer.last_add_special_tokens)

    def test_stream_metrics_separate_ttft_and_decode(self):
        responses = [
            SimpleNamespace(generation_tokens=count)
            for count in (1, 2, 3, 4)
        ]
        metrics, last = measure_stream(
            responses,
            clock=SequenceClock([10.0, 10.4, 10.5, 10.7, 11.0]),
        )

        self.assertEqual(last.generation_tokens, 4)
        self.assertAlmostEqual(metrics["ttft_seconds"], 0.4)
        self.assertAlmostEqual(metrics["decode_time_seconds"], 0.6)
        self.assertEqual(metrics["decode_tokens"], 3)
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 5.0)
        self.assertAlmostEqual(metrics["mean_inter_token_latency_ms"], 200.0)
        self.assertAlmostEqual(metrics["p50_inter_token_latency_ms"], 200.0)
        self.assertAlmostEqual(metrics["p95_inter_token_latency_ms"], 290.0)
        self.assertAlmostEqual(metrics["total_time_seconds"], 1.0)

    def test_empty_stream_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no tokens"):
            measure_stream([], clock=SequenceClock([0.0]))

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 4.0], 0.75), 3.0)


if __name__ == "__main__":
    unittest.main()
