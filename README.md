# HeadPatcher

Systematically identifies which attention heads in transformer language models are causally responsible for specific factual or relational knowledge, by applying head-level interventions and measuring output changes.

---

## Project Structure

```
HeadPatcher/
├── datasets/                   # Prompt datasets (JSONL)
│
├── answers/                    # Prompt answers (JSONL)
│
├── scripts/                    # Core pipeline
│   ├── run.py                  # Entry point: runs interventions, writes outputs
│   ├── core/
│   │   ├── config.py           # Model name, device, seed, dataset root
│   │   ├── data.py             # JSONL loader
│   │   ├── baseline.py         # Captures baseline logits/probs/hidden states
│   │   ├── metrics.py          # Per-prompt and per-head metric computation
│   │   ├── model.py            # HuggingFace model loader
│   │   └── io.py               # CSV writer
│   └── methods/
│       ├── zero_ablation.py    # Zeros the head's output dimensions
│       └── resampling_patch.py # Replaces head activations with a donor prompt's
│
├── analysis/                   # Post-hoc analysis scripts
│   └── select_heads.py         # Ranks heads by significance, selects top-K
│
└── outputs/                    # Generated results (gitignored)

```

---

## Quick Start

```bash
pip install -r requirement.txt

# Run with zero ablation
python scripts/run.py --method zero_ablation

# Select top-5 heads across all datasets, save results
python analysis/select_heads.py \
    --input outputs/zero_ablation \
    --method zero_ablation \
    --top-k 5 \
    --output analysis/resampling/

# Replace to json format
python analysis/build_annotation.py \
    --input analysis/zero_ablation \
    --output analysis/zero_ablation/annotations.json

```
Import json file to [visualization](https://headbb.vercel.app/)

---

## Pipeline

1. **Load model** (`core/model.py`) — defaults to `EleutherAI/pythia-1.4b`
2. **Load prompts** (`core/data.py`) — JSONL files grouped by `{category}/{source}`
3. **Compute baselines** (`core/baseline.py`) — forward pass capturing logits, probs, hidden states
4. **Intervene on each head** (`methods/`) — for every `(layer, head)` pair, apply method and re-run
5. **Compute metrics** (`core/metrics.py`) — probability deltas, KL divergence, entropy, token rank
6. **Write CSVs** (`core/io.py`) → `outputs/{method}/{category}/{source}/`

---

## Intervention Methods

| Method | Description | `requires_donor` |
|---|---|---|
| `resampling_patch` | Replaces head activations with a donor prompt's activations | Yes |
| `zero_ablation` | Zeros the head's output dimensions | No |

Both methods expose `intervene(model, baseline_items, layer, head)` and the `requires_donor` flag. New methods follow this interface.

---

## Dataset Format

```jsonl
{"prompt": "Habitat of camel is desert.\nHabitat of dolphin is"}
```

Files live at `datasets/{category}/{source}.jsonl`. The model predicts the next token after the final word.

---

## Output Format

### `summary_by_head.csv`

One row per head, sorted by `base_token_prob_delta_mean` (most negative first).

| Column | Description |
|---|---|
| `head` | Head label, e.g. `L15.H7` |
| `base_token_prob_delta_mean` | Mean change in baseline top-1 token probability after intervention |
| `base_token_prob_decrease_ratio` | Fraction of prompts where baseline token prob decreased |
| `base_token_changed_ratio` | Fraction of prompts where top-1 prediction changed |
| `base_token_logit_delta_mean` | Mean change in baseline token logit |
| `base_token_rank_post_mean` | Mean rank of baseline token after intervention |
| `entropy_delta_mean` | Mean change in output distribution entropy |
| `kl_divergence_mean` | Mean KL divergence D_KL(baseline ‖ modified) |
| `donor_token_prob_delta_mean` | *(resampling only)* Mean change in donor token probability |
| `donor_token_rank_pre_mean` | *(resampling only)* Mean rank of donor token before intervention |
| `donor_token_rank_post_mean` | *(resampling only)* Mean rank of donor token after intervention |

### `prompt_by_head.csv`

One row per `(head, prompt)` pair with the same metrics at per-prompt granularity.

---

## Head Selection (`analysis/select_heads.py`)

Ranks heads by `donor_token_rank_post_mean` ascending. Ties broken by `donor_token_logit_delta_mean` descending.

```bash
# Single dataset, top-5 (default)
python analysis/select_heads.py -i outputs/resampling_patch/capitals/europe/summary_by_head.csv

# All datasets, save results
python analysis/select_heads.py -i outputs/resampling_patch -o analysis/resampling/

# Top-10, zero_ablation
python analysis/select_heads.py -i outputs/zero_ablation -k 10 -m zero_ablation -o analysis/zero_ablation/
```

Output: `analysis/resampling/{category}/{source}/top5_heads.csv`

---

## Annotation Generation (`analysis/build_annotations.py`)

Scans all `top*_heads.csv` under the input directory and generates a structured JSON annotation file. Tags are derived from `{category}/{source}` folder names.

```bash
python analysis/build_annotations.py

# Custom paths
python analysis/build_annotations.py -i analysis/resampling -o analysis/annotations.json
```

Output: `analysis/annotations.json`
