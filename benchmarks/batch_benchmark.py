import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from benchmark_core import build_prompt_tokens, measure_stream


DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_INPUT_LENGTHS = [128, 256, 512, 1024]
DEFAULT_OUTPUT_LENGTHS = [64, 128, 256, 512]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Values must be positive integers.")
    return parsed


def run_inference(model, tokenizer, input_length, output_length):
    token_ids = build_prompt_tokens(tokenizer, input_length)

    mx.reset_peak_memory()
    mx.synchronize()
    metrics, last_response = measure_stream(
        stream_generate(
            model,
            tokenizer,
            prompt=token_ids,
            max_tokens=output_length,
        )
    )

    return {
        "requested_input_tokens": input_length,
        "requested_output_tokens": output_length,
        "input_tokens": last_response.prompt_tokens,
        "output_tokens": last_response.generation_tokens,
        "ttft_seconds": round(metrics["ttft_seconds"], 6),
        "decode_time_seconds": round(metrics["decode_time_seconds"], 6),
        "decode_tokens": metrics["decode_tokens"],
        "decode_tokens_per_second": round(
            metrics["decode_tokens_per_second"], 2
        ) if metrics["decode_tokens_per_second"] is not None else None,
        "mean_inter_token_latency_ms": round(
            metrics["mean_inter_token_latency_ms"], 3
        ) if metrics["mean_inter_token_latency_ms"] is not None else None,
        "p50_inter_token_latency_ms": round(
            metrics["p50_inter_token_latency_ms"], 3
        ) if metrics["p50_inter_token_latency_ms"] is not None else None,
        "p95_inter_token_latency_ms": round(
            metrics["p95_inter_token_latency_ms"], 3
        ) if metrics["p95_inter_token_latency_ms"] is not None else None,
        "elapsed_time_seconds": round(metrics["total_time_seconds"], 6),
        "prompt_tokens_per_second": round(last_response.prompt_tps, 2),
        "mlx_lm_generation_tokens_per_second": round(
            last_response.generation_tps, 2
        ),
        "peak_memory_gb": round(last_response.peak_memory, 4),
        "finish_reason": last_response.finish_reason,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an input/output length benchmark matrix using MLX-LM."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--input-lengths", nargs="+", type=positive_int,
        default=DEFAULT_INPUT_LENGTHS,
    )
    parser.add_argument(
        "--output-lengths", nargs="+", type=positive_int,
        default=DEFAULT_OUTPUT_LENGTHS,
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=positive_int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...", flush=True)
    model, tokenizer = load(args.model)
    all_results = []

    for input_length in args.input_lengths:
        for output_length in args.output_lengths:
            configuration = f"input={input_length}, output={output_length}"
            for warmup in range(args.warmups):
                print(
                    f"\nWarm-up {warmup + 1}/{args.warmups}: {configuration}",
                    flush=True,
                )
                run_inference(model, tokenizer, input_length, output_length)

            for repeat in range(args.repeats):
                print(
                    f"Testing {configuration}, "
                    f"run={repeat + 1}/{args.repeats}",
                    flush=True,
                )
                result = run_inference(
                    model, tokenizer, input_length, output_length
                )
                result.update({
                    "model": args.model,
                    "framework": "MLX-LM",
                    "run": repeat + 1,
                    "warmup": False,
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                all_results.append(result)

    print("\n=== Benchmark Summary ===")
    for input_length in args.input_lengths:
        for output_length in args.output_lengths:
            matching = [
                record for record in all_results
                if record["requested_input_tokens"] == input_length
                and record["requested_output_tokens"] == output_length
            ]
            ttft_ms = [record["ttft_seconds"] * 1000 for record in matching]
            decode_tps = [
                record["decode_tokens_per_second"] for record in matching
                if record["decode_tokens_per_second"] is not None
            ]
            print(
                f"Input {input_length}, output {output_length}: "
                f"TTFT={statistics.mean(ttft_ms):.1f} ms, "
                f"decode={statistics.mean(decode_tps):.2f} tok/s"
            )

    output_file = RESULTS_DIR / (
        "batch_benchmark_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
