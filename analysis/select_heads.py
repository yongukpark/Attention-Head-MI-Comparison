#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SORT_KEY = {
    "resampling_patch": lambda r: (float(r["donor_token_rank_post_mean"]), -float(r["donor_token_logit_delta_mean"])),
    "zero_ablation":    lambda r: float(r["base_token_prob_delta_mean"]),
}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def process_one(csv_path: Path, method: str, k: int, output: Path | None) -> None:
    rows = load_csv(csv_path)

    try:
        top = sorted(rows, key=SORT_KEY[method])[:k]
    except (KeyError, ValueError):
        print(f"[skip] {csv_path} — missing columns for method '{method}'", file=sys.stderr)
        return

    if output is not None:
        save_csv(output, top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--method", "-m", choices=list(SORT_KEY.keys()), default="resampling_patch")
    parser.add_argument("--top-k",  "-k", type=int, default=5)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if input_path.is_file():
        process_one(input_path, args.method, args.top_k, output_path)

    elif input_path.is_dir():
        csv_files = sorted(input_path.rglob("summary_by_head.csv"))
        if not csv_files:
            print(f"No summary_by_head.csv found under {input_path}", file=sys.stderr)
            sys.exit(1)
        for csv_file in csv_files:
            out = None
            if output_path is not None:
                rel = csv_file.relative_to(input_path)
                out = output_path / rel.parent / f"top{args.top_k}_heads.csv"
            process_one(csv_file, args.method, args.top_k, out)

    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
