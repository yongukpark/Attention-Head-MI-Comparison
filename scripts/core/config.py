from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np 
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT_DIR / "datasets"
DEFAULT_MODEL_NAME = "EleutherAI/pythia-1.4b"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
