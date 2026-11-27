"""Empathy top-up: keep only new candidates whose final user turn names a feeling.

The keyword list is frozen as empathy-keywords-v1. Do not edit it after screening starts.
Earlier rows must match the previous candidate file exactly, so the stream stays deterministic.

Run from the repo root (the pilot used the first 2,500 rows as --previous):
    head -n 2500 data/candidates-9000.jsonl > /tmp/previous-2500.jsonl
    python3 scripts/filter_empathy_topup.py --candidates data/candidates-9000.jsonl \\
        --previous /tmp/previous-2500.jsonl --out data/candidates-empathy-topup.jsonl
Then screen only the filtered file with frozen v3:
    python3 scripts/screen_wildchat.py --prompt-version opportunity-v3 --all \\
        --candidates data/candidates-empathy-topup.jsonl \\
        --report data/screening-review-empathy-topup.html
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

FILTER_VERSION = "empathy-keywords-v1"
PATTERN = (
    r"\b(i\s*(am|'m|m|feel|felt)\s+(so\s+|really\s+|very\s+)?(sad|depressed|anxious|nervous|scared|afraid|"
    r"worried|stressed|lonely|lost|confused|tired|exhausted|overwhelmed|frustrated|angry|upset|hopeless|hurt|"
    r"ashamed|embarrassed|heartbroken|miserable|drained)|feel(ing)?\s+(so\s+)?(sad|down|anxious|nervous|scared|"
    r"worried|stressed|lonely|lost|overwhelmed|frustrated|hopeless|bad|terrible)|afraid|scared|worried|anxiety|"
    r"depress\w*|lonely|heartbroken|grief|griev\w*|stress(ed|ful)|overwhelm\w*|frustrat\w*|crying|cried|nervous|"
    r"draining|drained|struggl\w*|hopeless|crisis|suicid\w*|panic\w*|devastat\w*|you don'?t understand|"
    r"i'?m not okay)\b"
)
PATTERN_SHA256 = "2fad1d4b51876cdc1e295fda73f8305f890857f803e1c8a5ff54773a72034486"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True, help="Extended candidate stream.")
    parser.add_argument("--previous", type=Path, required=True, help="Already-screened prefix of the stream.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(PATTERN.encode()).hexdigest() != PATTERN_SHA256:
        sys.exit(f"The {FILTER_VERSION} keyword list differs from the frozen version.")
    if args.out.exists():
        sys.exit(f"{args.out} already exists; choose a new name rather than overwrite it.")
    with args.previous.open(encoding="utf-8", newline="") as file:
        previous = [line.rstrip("\n") for line in file]
    regex = re.compile(PATTERN, re.IGNORECASE)
    try:
        _filter(args, previous, regex)
    except SystemExit:
        args.out.unlink(missing_ok=True)
        raise


def _filter(args, previous, regex):
    kept = scanned = 0
    index = -1
    with args.candidates.open(encoding="utf-8", newline="") as source, args.out.open("x", encoding="utf-8") as out:
        for index, line in enumerate(source):
            line = line.rstrip("\n")
            if index < len(previous):
                if line != previous[index]:
                    sys.exit(f"Row {index + 1} differs from {args.previous}; the stream is not an extension.")
                continue
            scanned += 1
            row = json.loads(line)
            if regex.search(row["messages"][-1]["content"]):
                row["topup_filter"] = FILTER_VERSION
                row["stream_position"] = index + 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    if index + 1 <= len(previous):
        sys.exit("The candidate file is not longer than the previous file.")
    print(f"Checked {len(previous)} earlier rows. Scanned {scanned} new rows; kept {kept}.", file=sys.stderr)


if __name__ == "__main__":
    main()
