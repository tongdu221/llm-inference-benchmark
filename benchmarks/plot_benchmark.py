import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TTFT, decode throughput, and end-to-end latency."
    )
    parser.add_argument(
        "--input", type=Path,
        help="Batch JSON file. Defaults to the newest compatible result.",
    )
    return parser.parse_args()


def load_records(requested_file=None):
    files = [requested_file] if requested_file else sorted(
        RESULTS_DIR.glob("batch_benchmark_*.json"), reverse=True
    )
    for result_file in files:
        with result_file.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if records and "ttft_seconds" in records[0]:
            return result_file, records
    raise FileNotFoundError(
        "No batch result containing TTFT/decode metrics was found."
    )


def mean_and_std(records, metric):
    values = [record[metric] for record in records if record[metric] is not None]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def grouped_records(records):
    groups = defaultdict(list)
    for record in records:
        key = (
            record["requested_input_tokens"],
            record["requested_output_tokens"],
        )
        groups[key].append(record)
    return groups


def save_line_chart(
    groups, input_lengths, output_lengths, metric, scale, ylabel,
    title, output_file, x_axis="input",
):
    plt.figure(figsize=(8.5, 5.2))
    series_values = output_lengths if x_axis == "input" else input_lengths
    x_values = input_lengths if x_axis == "input" else output_lengths

    for series in series_values:
        means = []
        stds = []
        for x_value in x_values:
            key = (
                (x_value, series) if x_axis == "input" else (series, x_value)
            )
            mean, std = mean_and_std(groups[key], metric)
            means.append(mean * scale)
            stds.append(std * scale)
        label = f"Output {series}" if x_axis == "input" else f"Input {series}"
        plt.errorbar(
            x_values, means, yerr=stds, marker="o", capsize=3,
            linewidth=1.8, label=label,
        )

    plt.xlabel(
        "Input Length (tokens)" if x_axis == "input"
        else "Output Length (tokens)"
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_values)
    plt.grid(True, alpha=0.25)
    plt.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    args = parse_args()
    result_file, records = load_records(args.input)
    groups = grouped_records(records)
    input_lengths = sorted({key[0] for key in groups})
    output_lengths = sorted({key[1] for key in groups})

    charts = [
        (
            "ttft_seconds", 1000, "TTFT (ms)",
            "Time to First Token vs Input Length",
            RESULTS_DIR / "ttft_by_input.png", "input",
        ),
        (
            "decode_tokens_per_second", 1, "Decode Throughput (tokens/s)",
            "Decode Throughput vs Input Length",
            RESULTS_DIR / "decode_throughput.png", "input",
        ),
        (
            "elapsed_time_seconds", 1, "End-to-End Latency (seconds)",
            "End-to-End Latency vs Output Length",
            RESULTS_DIR / "latency_by_output.png", "output",
        ),
    ]
    for metric, scale, ylabel, title, output_file, x_axis in charts:
        save_line_chart(
            groups, input_lengths, output_lengths, metric, scale, ylabel,
            title, output_file, x_axis,
        )
        print(f"Chart saved to: {output_file}")

    print(f"Loaded: {result_file.name}")


if __name__ == "__main__":
    main()
