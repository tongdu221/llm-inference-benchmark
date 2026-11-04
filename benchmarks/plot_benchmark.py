import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"

files = sorted(RESULTS_DIR.glob("batch_benchmark_*.json"))

if not files:
    raise FileNotFoundError("No batch benchmark results found.")

latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as file:
    records = json.load(file)

input_lengths = sorted(set(r["input_tokens"] for r in records))

means = []
stds = []

for length in input_lengths:
    speeds = [
        r["generation_tokens_per_second"]
        for r in records
        if r["input_tokens"] == length
    ]

    means.append(statistics.mean(speeds))
    stds.append(statistics.stdev(speeds))

plt.figure(figsize=(8, 5))

plt.errorbar(
    input_lengths,
    means,
    yerr=stds,
    marker="o",
    capsize=5,
    linewidth=2
)

plt.xlabel("Input Length (tokens)")
plt.ylabel("Generation Throughput (tokens/s)")
plt.title("Qwen3-0.6B Inference Benchmark (MLX, M2)")
plt.xticks(input_lengths)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = RESULTS_DIR / "generation_throughput.png"

plt.savefig(output_file, dpi=200)
plt.show()

print(f"Loaded: {latest_file.name}")
print(f"Chart saved to: {output_file}")