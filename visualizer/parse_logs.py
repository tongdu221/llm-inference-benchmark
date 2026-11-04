import re

def parse_log(line):
    pattern = (
        r"iter=(\d+).*?"
        r"phase=(\w+).*?"
        r"batch=(\d+).*?"
        r"kv_blocks=(\d+)/(\d+)"
    )

    match = re.search(pattern, line)

    if not match:
        return None

    iteration, phase, batch, used, total = match.groups()

    return {
        "iteration": int(iteration),
        "phase": phase,
        "batch": int(batch),
        "used_blocks": int(used),
        "total_blocks": int(total),
    }


if __name__ == "__main__":
    sample = (
        "[Task3] iter=243 phase=decode batch=1 "
        "running=1 waiting=0 decode_tokens=1 kv_blocks=2/2532"
    )

    print(parse_log(sample))