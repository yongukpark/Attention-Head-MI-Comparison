#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated and will be removed in version 1\.0\.0\..*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained\.",
    category=UserWarning,
)

from core.baseline import prepare_baseline
from core.config import (
    DATASET_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    DEVICE,
    configure_reproducibility,
)

from core.data import load_prompt_items
from core.io import save_csv
from core.metrics import compute_head_summary, compute_prompt_metrics
from core.model import load_model
from methods import resampling_patch, zero_ablation
from tqdm import tqdm

METHODS = {
    "resampling_patch": resampling_patch,
    "zero_ablation": zero_ablation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Head intervention mining.")
    parser.add_argument("--method", choices=list(METHODS.keys()), default="resampling_patch")
    parser.add_argument(
        "--prompt-path",
        "--prompt-file",
        dest="prompt_path",
        default=str(DATASET_ROOT),
        help="JSONL file or directory to scan recursively (default: datasets root).",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    args = parser.parse_args()

    method = METHODS[args.method]
    configure_reproducibility(DEFAULT_SEED)

    out_dir = Path(args.output_dir)
    prompt_items = load_prompt_items(DATASET_ROOT, args.prompt_path.strip())

    # Group prompts by (category, source_file_stem) → determines output directory structure
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for item in prompt_items:
        category    = item["category"]
        source_name = Path(item["source_file"]).stem
        by_bucket.setdefault((category, source_name), []).append(item)

    print(f"Method: {args.method}")
    print(f"Loading model: {DEFAULT_MODEL_NAME} on {DEVICE} ...")
    model, tokenizer = load_model(DEFAULT_MODEL_NAME, DEVICE)
    print("Model loaded.")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    total_heads = n_layers * n_heads

    for bk in sorted(by_bucket.keys()):
        category, source_name = bk
        items = by_bucket[bk]

        min_prompts = 2 if method.requires_donor else 1

        if len(items) < min_prompts:
            print(f"  skip '{category}/{source_name}': need at least {min_prompts} prompts")
            continue

        print(f"\n[{category}/{source_name}] {len(items)} prompts")
        baseline_items = prepare_baseline(model, tokenizer, DEVICE, items)

        summary_rows: list[dict] = []
        prompt_rows: list[dict] = []

        head_iter = ((l, h) for l in range(n_layers) for h in range(n_heads))
        for layer, head in tqdm(head_iter, total=total_heads, desc=f"  {category}/{source_name}", unit="head"):
            head_label = f"L{layer}.H{head}"
            results = method.intervene(model, baseline_items, layer, head)

            head_prompt_metrics = []
            for i, result in enumerate(results):
                base = baseline_items[i]
                donor = baseline_items[result["donor_index"]] if "donor_index" in result else None
                metrics = compute_prompt_metrics(base, result["modified_logits"], result["modified_probs"], donor)

                prompt_rows.append({
                    "head": head_label,
                    "prompt_index": i,
                    "prompt": base["prompt"],
                    **metrics,
                })
                head_prompt_metrics.append(metrics)

            summary = compute_head_summary(head_prompt_metrics)
            summary_rows.append({"head": head_label, **summary})

        # Sort summary by base_token_prob_delta_mean (most negative first)
        summary_rows.sort(key=lambda r: r["base_token_prob_delta_mean"])

        bucket_dir = out_dir / args.method / category / source_name
        save_csv(bucket_dir / "summary_by_head.csv", summary_rows)
        save_csv(bucket_dir / "prompt_by_head.csv", prompt_rows)
        print(f"  saved: {bucket_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
