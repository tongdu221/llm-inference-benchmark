
from pathlib import Path

import matplotlib.pyplot as plt
from parse_logs import parse_log

# Locate files relative to this script
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "sample.log"
OUTPUT_FILE = BASE_DIR / "kv_cache_observations.png"

# Read and parse the log file
records = []

with LOG_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        record = parse_log(line)
        if record is not None:
            records.append(record)

if not records:
    raise ValueError("No valid log records found.")

# Sort records by iteration
records.sort(key=lambda record: record["iteration"])

iterations = [record["iteration"] for record in records]
used_blocks = [record["used_blocks"] for record in records]

# Plot KV-cache usage
plt.figure(figsize=(8, 4.5))

plt.step(
    iterations,
    used_blocks,
    where="post",
    linewidth=2,
    label="KV-cache blocks"
)

plt.scatter(iterations, used_blocks, zorder=3)

plt.xticks(iterations)
plt.yticks(range(max(used_blocks) + 1))

plt.xlabel("Iteration")
plt.ylabel("Used KV-cache Blocks")
plt.title("KV-cache Allocation and Release (Selected Observations)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=200)
plt.show()

print(f"Parsed {len(records)} log records.")
print(f"Chart saved to: {OUTPUT_FILE}")
print("Iterations:", iterations)
print("Used blocks:", used_blocks)
print("Parsed records:", len(records))