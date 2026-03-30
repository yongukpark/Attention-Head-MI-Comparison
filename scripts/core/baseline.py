from __future__ import annotations

import torch
from tqdm import tqdm

from core.metrics import entropy
from core.model import get_dense_module, make_padded_batch


def prepare_baseline(model, tokenizer, device: torch.device, prompt_items: list[dict], batch_size: int = 32) -> list[dict]:
    n_layers = model.config.num_hidden_layers
    pad_token_id = tokenizer.pad_token_id

    # Pre-tokenize all prompts (keep original unpadded tensors for later use in methods)
    all_input_ids = [
        tokenizer(item["prompt"], return_tensors="pt").input_ids
        for item in prompt_items
    ]

    baseline_items: list[dict] = []
    pbar = tqdm(total=len(prompt_items), desc="Baseline", unit="prompt")

    for batch_start in range(0, len(prompt_items), batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        batch_items = prompt_items[batch_slice]
        batch_ids = all_input_ids[batch_slice]

        padded, attn_mask = make_padded_batch(batch_ids, pad_token_id, device)  # [B, max_len]

        # Cache each layer's last-token hidden state for all items in the batch
        cached: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in range(n_layers):
            def build_hook(li: int):
                def hook(_module, inputs):
                    # inputs[0]: [B, seq, hidden] → keep last token for all items
                    cached[li] = inputs[0][:, -1].detach().clone()  # [B, hidden]
                    return inputs
                return hook
            handles.append(get_dense_module(model, layer_idx).register_forward_pre_hook(build_hook(layer_idx)))

        with torch.inference_mode():
            logits = model(padded, attention_mask=attn_mask).logits  # [B, seq, vocab]
            last_logits = logits[:, -1, :]                           # [B, vocab]
            last_probs = torch.softmax(last_logits, dim=-1)
        for h in handles:
            h.remove()

        for bi, (item, orig_ids) in enumerate(zip(batch_items, batch_ids)):
            hidden_by_layer = {li: cached[li][bi].cpu() for li in cached}
            llogits = last_logits[bi]
            lprobs = last_probs[bi]
            top1_id = int(torch.argmax(lprobs).item())
            baseline_items.append({
                **item,
                "input_ids": orig_ids,  # original unpadded ids (CPU); methods will move to device
                "pad_token_id": pad_token_id,
                "baseline_logits": llogits.cpu(),
                "baseline_probs": lprobs.cpu(),
                "baseline_top1_id": top1_id,
                "baseline_top1_token": tokenizer.decode([top1_id]),
                "baseline_top1_logit": float(llogits[top1_id].item()),
                "baseline_top1_prob": float(lprobs[top1_id].item()),
                "baseline_entropy": entropy(lprobs),
                "hidden_by_layer": hidden_by_layer,
            })

        pbar.update(len(batch_items))

    pbar.close()
    return baseline_items
