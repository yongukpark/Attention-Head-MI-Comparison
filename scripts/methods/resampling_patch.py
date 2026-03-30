from __future__ import annotations

import torch

from core.model import forward_batch_last_token, get_dense_module, make_padded_batch


def intervene(model, baseline_items, layer, head):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    dense = get_dense_module(model, layer)
    start = head * head_dim
    end = start + head_dim

    n = len(baseline_items)
    donor_indices = [(i + 1) % n for i in range(n)]

    # Pre-stack donor vectors: [B, head_dim] on CPU, move to device in hook
    donor_vecs = torch.stack([
        baseline_items[donor_indices[i]]["hidden_by_layer"][layer][start:end]
        for i in range(n)
    ])  # [B, head_dim]

    def hook(_, inputs):
        hidden = inputs[0].clone()
        hidden[:, -1, start:end] = donor_vecs.to(hidden.device)
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
        {
            "modified_logits": logits_batch[i],
            "modified_probs": probs_batch[i],
            "donor_index": donor_indices[i],
        }
        for i in range(n)
    ]


requires_donor = True
