import torch
import torch.nn.functional as F
from typing import Dict, Optional


def compute_token_level_kl(
    logprobs_current: torch.FloatTensor,
    logprobs_ref: torch.FloatTensor,
    kl_penalty: str = "kl",
) -> torch.FloatTensor:
    """
    Compute KL divergence using token-level log probabilities.

    Mirrors verl.trainer.ppo.core_algos.kl_penalty for tokenwise values.

    Args:
        logprobs_current: Log probabilities of selected tokens from current model [batch_size, seq_len]
        logprobs_ref: Log probabilities of selected tokens from reference model [batch_size, seq_len]
        kl_penalty: Type of KL penalty ("kl", "abs", "mse", "low_var_kl", aliases: "k1", "k2", "k3")

    Returns:
        KL divergence per token [batch_size, seq_len]
    """
    if kl_penalty in ("kl", "k1"):
        return logprobs_current - logprobs_ref

    if kl_penalty == "abs":
        return (logprobs_current - logprobs_ref).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprobs_current - logprobs_ref).square()

    if kl_penalty in ("low_var_kl", "k3"):
        kl = logprobs_ref - logprobs_current
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    raise ValueError(f"Unknown kl_penalty type: {kl_penalty}")


def extract_token_log_probs(
    logits: torch.FloatTensor,
    token_ids: torch.LongTensor,
) -> torch.FloatTensor:
    """
    Extract log probabilities for specific tokens from logits.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        token_ids: Token IDs to extract log probs for [batch_size, seq_len]

    Returns:
        Log probabilities of the specified tokens [batch_size, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def compute_sequence_level_kl(
    logprobs_current: torch.FloatTensor,
    logprobs_ref: torch.FloatTensor,
    response_mask: Optional[torch.FloatTensor] = None,
    kl_penalty: str = "kl",
) -> Dict[str, float]:
    """
    Compute sequence-level KL metrics from token log probabilities.

    Args:
        logprobs_current: Log probs of tokens from current model [batch_size, seq_len]
        logprobs_ref: Log probs of tokens from reference model [batch_size, seq_len]
        response_mask: Mask for response tokens (1 for response, 0 for prompt/padding)
        kl_penalty: Type of KL penalty to use

    Returns:
        Dictionary with KL metrics
    """
    kl = compute_token_level_kl(logprobs_current, logprobs_ref, kl_penalty)

    if response_mask is not None:
        kl = kl * response_mask
        num_tokens = response_mask.sum(dim=-1)
        num_tokens = torch.clamp(num_tokens, min=1)
    else:
        num_tokens = torch.ones(kl.shape[0], device=kl.device) * kl.shape[1]

    kl_sum_per_seq = kl.sum(dim=-1)
    kl_mean_per_seq = kl_sum_per_seq / num_tokens
    kl_max_per_seq = kl.max(dim=-1).values

    return {
        "kl_mean": kl_mean_per_seq.mean().item(),
        "kl_sum": kl_sum_per_seq.mean().item(),
        "kl_max": kl_max_per_seq.mean().item(),
        "kl_penalty_type": kl_penalty,
    }


