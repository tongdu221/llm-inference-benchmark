import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm import load, stream_generate


MODEL = "mlx-community/Qwen3-0.6B-4bit"
INPUT_LENGTHS = [128, 256, 512, 1024]
OUTPUT_LENGTH = 128
REPEATS = 3

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_inference(model, tokenizer, input_length):
    text = "hello "
    base_tokens = tokenizer.encode(
        text, add_special_tokens=False
    )

    token_ids = (
        base_tokens * ((input_length // len(base_tokens)) + 1)
    )[:input_length]

    start_time = time.perf_counter()
    last_response = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt=token_ids,
        max_tokens=OUTPUT_LENGTH
    ):
        last_response = response

    elapsed = time.perf_counter() - start_time

    if last_response is None:
        raise RuntimeError("No tokens generated.")

    return {
        "input_tokens": last_response.prompt_tokens,
        "output_tokens": last_response.generation_tokens,
        "elapsed_time_seconds": round(elapsed, 4),
        "prompt_tokens_per_second": round(
            last_response.prompt_tps, 2
        ),
        "generation_tokens_per_second": round(
            last_response.generation_tps, 2
        ),
        "peak_memory_gb": round(
            last_response.peak_memory, 4
        ),
        "finish_reason": last_response.finish_reason
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...", flush=True)
    model, tokenizer = load(MODEL)

    all_results = []

    for input_length in INPUT_LENGTHS:
        print(
            f"\nWarm-up: input={input_length}",
            flush=True
        )

        run_inference(model, tokenizer, input_length)

        for repeat in range(REPEATS):
            print(
                f"Testing input={input_length}, "
                f"run={repeat + 1}/{REPEATS}",
                flush=True
            )

            result = run_inference(
                model, tokenizer, input_length
            )

            result.update({
                "model": MODEL,
                "framework": "MLX-LM",
                "run": repeat + 1,
                "warmup": False,
                "system": platform.system(),
                "machine": platform.machine(),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat()
            })

            all_results.append(result)

    print("\n=== Benchmark Summary ===")

    for input_length in INPUT_LENGTHS:
        speeds = [
            r["generation_tokens_per_second"]
            for r in all_results
            if r["input_tokens"] == input_length
        ]

        print(
            f"Input {input_length}: "
            f"mean={statistics.mean(speeds):.2f} tok/s, "
            f"std={statistics.stdev(speeds):.2f}"
        )

    output_file = RESULTS_DIR / (
        "batch_benchmark_"
        + datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".json"
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()