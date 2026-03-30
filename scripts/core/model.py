from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: torch.device):
    # Load in float16 on CUDA to halve memory usage and speed up inference
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def make_padded_batch(
    input_ids_list: list[torch.Tensor], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of [1, seq_len] input_ids tensors into a [B, max_len] batch.

    Returns:
        input_ids:      [B, max_len] — left-padded token ids
        attention_mask: [B, max_len] — 0 for pad positions, 1 for real tokens
    """
    max_len = max(ids.shape[-1] for ids in input_ids_list)
    rows, masks = [], []
    for ids in input_ids_list:
        seq_len = ids.shape[-1]
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad = torch.full((1, pad_len), pad_token_id, dtype=ids.dtype, device=device)
            ids = torch.cat([pad, ids.to(device)], dim=-1)
            mask = torch.cat([
                torch.zeros(1, pad_len, dtype=torch.long, device=device),
                torch.ones(1, seq_len, dtype=torch.long, device=device),
            ], dim=-1)
        else:
            ids = ids.to(device)
            mask = torch.ones(1, max_len, dtype=torch.long, device=device)
        rows.append(ids)
        masks.append(mask)
    return torch.cat(rows, dim=0), torch.cat(masks, dim=0)  # [B, max_len] each


def forward_last_token(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        logits = model(input_ids).logits
        last_logits = logits[0, -1]
        last_probs = torch.softmax(last_logits, dim=-1)
    return last_logits, last_probs


def forward_batch_last_token(
    model, input_ids_batch: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass on [B, seq] batch; return (last_logits, last_probs) both [B, vocab]."""
    with torch.inference_mode():
        logits = model(input_ids_batch, attention_mask=attention_mask).logits  # [B, seq, vocab]
        last_logits = logits[:, -1, :]  # [B, vocab]
        last_probs = torch.softmax(last_logits, dim=-1)
    return last_logits, last_probs


def get_dense_module(model, layer: int):
    return model.gpt_neox.layers[layer].attention.dense
