from __future__ import annotations

import torch


def entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    p = probs.float().clamp_min(eps)  # cast to float32: float16 flushes 1e-12 to 0
    return float((-(p * torch.log(p))).sum().item())


def compute_prompt_metrics(
    base_item: dict,
    modified_logits: torch.Tensor,
    modified_probs: torch.Tensor,
    donor_item: dict | None = None,
) -> dict:
    base_top1_id = base_item["baseline_top1_id"]
    baseline_probs = base_item["baseline_probs"]
    baseline_logits = base_item["baseline_logits"]
    eps = 1e-12

    modified_logits = modified_logits.cpu()
    modified_probs = modified_probs.cpu()

    mod_entropy = entropy(modified_probs)
    base_entropy = base_item["baseline_entropy"]

    # rank = number of tokens with higher probability than the target token + 1
    base_rank_post = int((modified_probs  > modified_probs[base_top1_id]).sum().item()) + 1

    # KL divergence D_KL(baseline || modified): how much the modified distribution diverges from baseline
    p = baseline_probs.float().clamp_min(eps)  # cast to float32: float16 flushes 1e-12 to 0
    q = modified_probs.float().clamp_min(eps)
    kl = float((p * (torch.log(p) - torch.log(q))).sum().item())

    row = {
        "base_token_prob_delta": float(modified_probs[base_top1_id].item() - baseline_probs[base_top1_id].item()),
        "base_token_logit_delta": float(modified_logits[base_top1_id].item()) - base_item["baseline_top1_logit"],
        "base_token_changed": modified_probs.argmax().item() != base_top1_id,
        "base_token_rank_post": base_rank_post,
        "baseline_entropy": base_entropy,
        "modified_entropy": mod_entropy,
        "entropy_delta": mod_entropy - base_entropy,
        "kl_divergence": kl,
    }

    if donor_item is not None:
        donor_top1_id = donor_item["baseline_top1_id"]
        row["donor_token_prob_delta"]  = float(modified_probs[donor_top1_id].item() - baseline_probs[donor_top1_id].item())
        row["donor_token_logit_delta"] = float(modified_logits[donor_top1_id].item() - baseline_logits[donor_top1_id].item())
        row["donor_token_rank_pre"]    = int((baseline_probs > baseline_probs[donor_top1_id]).sum().item()) + 1
        row["donor_token_rank_post"]   = int((modified_probs  > modified_probs[donor_top1_id]).sum().item()) + 1

    return row


def compute_head_summary(prompt_metrics: list[dict]) -> dict:
    n = len(prompt_metrics)
    prob_deltas = [r["base_token_prob_delta"] for r in prompt_metrics]

    summary = {
        "prompt_count": n,
        "base_token_prob_delta_mean":     sum(prob_deltas) / n,
        "base_token_prob_decrease_ratio": sum(1 for d in prob_deltas if d < 0) / n,
        "base_token_changed_ratio":       sum(1 for r in prompt_metrics if r["base_token_changed"]) / n,
        "base_token_logit_delta_mean":    sum(r["base_token_logit_delta"] for r in prompt_metrics) / n,
        "base_token_rank_post_mean":      round(sum(r["base_token_rank_post"]   for r in prompt_metrics) / n),
        "entropy_delta_mean":             sum(r["entropy_delta"]          for r in prompt_metrics) / n,
        "kl_divergence_mean":             sum(r["kl_divergence"]          for r in prompt_metrics) / n,
    }

    if "donor_token_prob_delta" in prompt_metrics[0]:
        summary["donor_token_prob_delta_mean"]  = sum(r["donor_token_prob_delta"]  for r in prompt_metrics) / n
        summary["donor_token_logit_delta_mean"] = sum(r["donor_token_logit_delta"] for r in prompt_metrics) / n
        summary["donor_token_rank_pre_mean"]    = round(sum(r["donor_token_rank_pre"]    for r in prompt_metrics) / n)
        summary["donor_token_rank_post_mean"]   = round(sum(r["donor_token_rank_post"]   for r in prompt_metrics) / n)

    return summary
