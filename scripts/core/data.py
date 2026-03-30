from __future__ import annotations

import json
import re
from pathlib import Path

from core.config import DATASET_ROOT


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Only .jsonl is supported: {path}")
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("prompt"), str) and row["prompt"].strip():
            prompts.append(row["prompt"].strip())
        elif isinstance(row, str) and row.strip():
            prompts.append(row.strip())
    return prompts


def load_prompt_items(dataset_root: Path, prompt_file: str | None = None) -> list[dict]:
    items: list[dict] = []

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found or invalid: {dataset_root}")

    target_path = _resolve_prompt_file_path(prompt_file) if prompt_file else dataset_root.resolve()
    files = _collect_prompt_files(target_path)

    for fp in files:
        prompts = load_prompts(fp)
        if not prompts:
            continue
        category = _infer_category(fp, dataset_root.resolve(), target_path)
        for prompt in prompts:
            items.append({"prompt": prompt, "source_file": str(fp), "category": category})

    if not items:
        raise ValueError(f"No prompts found from path: {target_path}")
    return items


def _resolve_prompt_file_path(prompt_file: str) -> Path:
    prompt_path = Path(prompt_file).expanduser()
    if prompt_path.exists():
        return prompt_path.resolve()
    candidate = DATASET_ROOT / prompt_path
    if candidate.exists():
        return candidate.resolve()
    return prompt_path


def _collect_prompt_files(target_path: Path) -> list[Path]:
    if not target_path.exists():
        raise FileNotFoundError(f"Prompt path not found: {target_path}")
    if target_path.is_file():
        if target_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Only .jsonl is supported: {target_path}")
        return [target_path]
    if not target_path.is_dir():
        raise ValueError(f"Prompt path must be a file or directory: {target_path}")
    return sorted(target_path.rglob("*.jsonl"))


def _infer_category(fp: Path, dataset_root: Path, target_path: Path) -> str:
    for base in (dataset_root, target_path if target_path.is_dir() else target_path.parent):
        try:
            rel = fp.relative_to(base)
            if len(rel.parts) >= 2 and rel.parts[0] == "by_category":
                return rel.parts[1]
            if len(rel.parts) >= 2:
                return rel.parts[0]
        except ValueError:
            continue
    return fp.parent.name or "uncategorized"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-") or "uncategorized"
