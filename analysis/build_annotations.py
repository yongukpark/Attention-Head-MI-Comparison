#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

MODEL_NAME = "Pythia-1.4B"
NUM_LAYERS = 24
NUM_HEADS = 16

def parse_head_label(label: str) -> tuple[int, int]:
    layer_part, head_part = label.split(".")
    return int(layer_part[1:]), int(head_part[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default="analysis/resampling")
    parser.add_argument("--output", "-o", default="analysis/annotations.json")
    args = parser.parse_args()

    input_dir   = Path(args.input)
    output_path = Path(args.output)

    # head_key → {tag → best_rank}
    head_best: dict[str, dict[str, int]] = {}

    for csv_path in sorted(input_dir.rglob("top*_heads.csv")):
        rel      = csv_path.relative_to(input_dir)
        category = rel.parts[0]
        source   = rel.parts[1]
        tag      = f"{category}/{source}"

        with open(csv_path, newline="") as f:
            for rank, row in enumerate(csv.DictReader(f), start=1):
                layer, head = parse_head_label(row["head"])
                key = f"L{layer}H{head}"
                if key not in head_best:
                    head_best[key] = {}
                if tag not in head_best[key] or rank < head_best[key][tag]:
                    head_best[key][tag] = rank

    # For each tag, re-rank heads by their best_rank → ordinal 1, 2, 3...
    tag_to_entries: dict[str, list[tuple[str, int]]] = {}
    for key, tag_ranks in head_best.items():
        for tag, best_rank in tag_ranks.items():
            tag_to_entries.setdefault(tag, []).append((key, best_rank))

    tag_ordinal: dict[str, dict[str, int]] = {}
    for tag, entries in tag_to_entries.items():
        for ordinal, (key, _) in enumerate(sorted(entries, key=lambda x: x[1]), start=1):
            tag_ordinal.setdefault(tag, {})[key] = ordinal

    # Build annotations
    annotations: dict = {}
    all_tags: set[str] = set()

    for key in sorted(head_best):
        layer_str, head_str = key[1:].split("H")
        tags = sorted(head_best[key].keys())
        descriptions = {tag: f"({tag_ordinal[tag][key]})" for tag in tags}
        annotations[key] = {
            "layer":        int(layer_str),
            "head":         int(head_str),
            "tags":         tags,
            "descriptions": descriptions,
        }
        all_tags.update(tags)

    # Preserve createdAt if file already exists
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        created_at = existing.get("createdAt", created_at)

    result = {
        "modelName":   MODEL_NAME,
        "numLayers":   NUM_LAYERS,
        "numHeads":    NUM_HEADS,
        "annotations": annotations,
        "tags":        sorted(all_tags),
        "createdAt":   created_at,
        "updatedAt":   datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(annotations)} head annotations → {output_path}")


if __name__ == "__main__":
    main()
