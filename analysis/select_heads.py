#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SORT_KEY = {
    "resampling_patch": ("donor_token_rank_post_mean", True),
    "zero_ablation":    ("base_token_prob_delta_mean", True),
}

TIEBREAK = {
    "resampling_patch": ("donor_token_logit_delta_mean", False),
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
    sort_col, ascending = SORT_KEY[method]

    if not rows or sort_col not in rows[0]:
        print(f"[skip] {csv_path} — no '{sort_col}' column", file=sys.stderr)
        return

    tb_col, tb_desc = TIEBREAK.get(method, (None, None))

    def sort_key(r):
        primary = float(r[sort_col]) * (1 if ascending else -1)
        secondary = -float(r[tb_col]) if (tb_col and tb_desc and r.get(tb_col) not in (None, "")) else 0
        return (primary, secondary)

    top = sorted(
        [r for r in rows if r.get(sort_col) not in (None, "")],
        key=sort_key,
    )[:k]

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
