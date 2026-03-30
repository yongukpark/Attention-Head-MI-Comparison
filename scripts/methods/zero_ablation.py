from __future__ import annotations

import torch

from core.model import forward_batch_last_token, get_dense_module, make_padded_batch


def intervene(model, baseline_items, layer, head):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    dense = get_dense_module(model, layer)
    start = head * head_dim
    end = start + head_dim

    def hook(_module, inputs):
        hidden = inputs[0].clone()
        hidden[:, -1, start:end] = 0.0
        return (hidden,)

    pad_token_id = baseline_items[0]["pad_token_id"]
    device = next(model.parameters()).device
    input_ids_batch, attn_mask = make_padded_batch(
        [item["input_ids"] for item in baseline_items], pad_token_id, device
    )

    handle = dense.register_forward_pre_hook(hook)
    logits_batch, probs_batch = forward_batch_last_token(model, input_ids_batch, attn_mask)
    handle.remove()

    return [
        {"modified_logits": logits_batch[i], "modified_probs": probs_batch[i]}
        for i in range(len(baseline_items))
    ]


requires_donor = False
