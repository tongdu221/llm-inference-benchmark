import argparse
import json
import platform
from datetime import datetime, timezone

import mlx.core as mx
from mlx_lm import load, stream_generate
from pathlib import Path

from benchmark_core import build_prompt_tokens, measure_stream


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

    token_ids = build_prompt_tokens(tokenizer, args.input_len)
    
    print("Running inference...")

    
    mx.reset_peak_memory()

    mx.synchronize()
    metrics, last_response = measure_stream(
        stream_generate(
            model,
            tokenizer,
            prompt=token_ids,
            max_tokens=args.output_len
        )
    )

    result = {
        "model": args.model,
        "framework": "MLX-LM",
        "requested_input_tokens": args.input_len,
        "requested_output_tokens": args.output_len,
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
        "prompt_tokens_per_second": round(
            last_response.prompt_tps, 2
        ),
        "mlx_lm_generation_tokens_per_second": round(
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
