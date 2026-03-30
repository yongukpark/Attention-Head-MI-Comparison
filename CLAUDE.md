# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Install dependencies
pip install -r requirement.txt

# Run with default settings (zero_ablation, all datasets)
python scripts/run.py

# Run with specific method
python scripts/run.py --method resampling_patch

# Run on a specific prompt file
python scripts/run.py --prompt-path datasets/animals/habitat.jsonl

# Specify output directory
python scripts/run.py --output-dir results/
```

No build system, test suite, or linter is configured — this is a pure Python research project.

## Architecture

HeadPatcher analyzes which attention heads in transformer language models influence predictions. It does this by systematically applying interventions (zeroing or patching activations) to each head and measuring the effect on model output.

**Pipeline (`scripts/run.py`):**
1. Load model (`core/model.py`) — defaults to `EleutherAI/pythia-1.4b`
2. Load prompt datasets from `datasets/` (`core/data.py`) — JSONL files grouped by category/source
3. Compute baselines for all prompts (`core/baseline.py`) — captures hidden states via forward hooks
4. For each (layer, head) combination, apply an intervention method and measure output change
5. Compute metrics (`core/metrics.py`) — probability deltas, KL divergence, entropy, token rank
6. Write CSVs to `outputs/{category}/{source}/` (`core/io.py`)

**Intervention methods (`scripts/methods/`):**
- `zero_ablation.py` — zeros the head's dimensions; `requires_donor = False`
- `resampling_patch.py` — replaces head activations with a donor prompt's activations; `requires_donor = True`

Both methods expose an `intervene()` function and the `requires_donor` flag. New methods follow this same interface.

**Hook mechanism:** Interventions operate via PyTorch forward hooks on the attention output (dense projection) layer, modifying hidden states in-place during inference.

## Key Configuration

`scripts/core/config.py` controls:
- `DEFAULT_MODEL_NAME` — model to load (Hugging Face model ID)
- `DEFAULT_SEED` — reproducibility seed (default: 42); also disables CUDA non-determinism and TF32
- `DEVICE` — auto-detects CUDA, falls back to CPU
- `DATASET_ROOT` — resolves to `datasets/` relative to the repo root

## Dataset Format

JSONL files under `datasets/{category}/{source}.jsonl`, each line:
```json
{"prompt": "Habitat of camel is desert.\nHabitat of dolphin is"}
```

The model predicts the next token after the last word in the prompt. Category and source name are inferred from the file path.

## Output Format

- `outputs/{category}/{source}/summary_by_head.csv` — per-head aggregated metrics across all prompts
- `outputs/{category}/{source}/prompt_by_head.csv` — per-prompt per-head intervention results

Key summary columns: `base_token_prob_delta_mean`, `base_token_changed_ratio`, `kl_divergence_mean`, `entropy_delta_mean`.
