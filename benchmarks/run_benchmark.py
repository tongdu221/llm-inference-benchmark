import argparse
import json
import platform
import time
from datetime import datetime, timezone

import mlx.core as mx
from mlx_lm import load, stream_generate
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="LLM inference benchmark using MLX"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-0.6B-4bit"
    )

    parser.add_argument(
        "--input-len",
        type=int,
        default=128
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=128
    )

    args = parser.parse_args()

    if args.input_len <= 0 or args.output_len <= 0:
        parser.error("Token lengths must be positive.")

    print("Loading model...")
    model, tokenizer = load(args.model)

    # Construct a prompt with exactly the requested token count.
    text = "hello "
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    while len(token_ids) < args.input_len:
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))

    token_ids = token_ids[:args.input_len]
    actual_input_tokens = len(token_ids)
    
    print("Running inference...")

    
    mx.reset_peak_memory()

    start_time = time.perf_counter()

    last_response = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt=token_ids,
        max_tokens=args.output_len
    ):
        last_response = response

    elapsed_time = time.perf_counter() - start_time

    if last_response is None:
        raise RuntimeError("Model generated no tokens.")

    actual_output_tokens = last_response.generation_tokens

    result = {
        "model": args.model,
        "framework": "MLX-LM",
        "input_tokens": last_response.prompt_tokens,
        "output_tokens": actual_output_tokens,
        "elapsed_time_seconds": round(elapsed_time, 4),
        "prompt_tokens_per_second": round(
            last_response.prompt_tps, 2
        ),
        "generation_tokens_per_second": round(
            last_response.generation_tps, 2
        ),
        "peak_memory_gb": round(
            last_response.peak_memory, 4
        ),
        "finish_reason": last_response.finish_reason,
        "system": platform.system(),
        "machine": platform.machine(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    print(json.dumps(result, indent=2))
    
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"benchmark_{args.input_len}_{args.output_len}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
    )

    output_path = results_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()