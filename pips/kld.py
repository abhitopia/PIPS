import math
import torch
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical, RelaxedOneHotCategorical
from torch.nn import functional as F
from typing import NamedTuple

class KLDLosses(NamedTuple):
    overall_kl: torch.Tensor 
    mutual_info: torch.Tensor = torch.tensor(0.0)
    total_correlation: torch.Tensor = torch.tensor(0.0)
    dimension_wise_kl: torch.Tensor = torch.tensor(0.0)

def compute_decomposed_kld(
    log_alpha, 
    running_q_marginals=None, 
    momentum=0.9, 
    eps=1e-10, 
    reduction="mean"
):
    """
    Compute per-sample estimates of KL, MI, and then decompose 
    Total Correlation (TC = KL - MI - DWKL) using an EMA for the aggregated posterior.
    
    The per-sample KL and MI are computed by summing over the latent dimensions and codebook entries,
    and then TC is computed per sample before applying the reduction.
    
    Args:
        log_alpha: Tensor of shape [B, N, C] (unnormalized logits).
        running_q_marginals: Tensor of shape [N, C] representing the running aggregated marginals.
                             If None, it is initialized with the current batch average.
        momentum: EMA momentum.
        eps: Small constant for numerical stability.
        reduction: One of 'sum', 'mean', or 'batchmean'.  
            'sum': Sum over the batch.
            'mean' or 'batchmean': Average over the batch.
    
    Returns:
        KLDResults: A named tuple containing:
            - mutual_info: Scalar tensor (after reduction) for the estimated mutual information.
            - total_correlation: Scalar tensor (after reduction) for the estimated total correlation.
            - dimension_wise_kl: Scalar tensor for the estimated dimension-wise KL (global).
            - overall_kl: Scalar tensor (after reduction) for the overall KL divergence.
        running_q_marginals: Updated running aggregated marginals (Tensor of shape [N, C]).
    """
    B, N, C = log_alpha.shape

    # Compute soft probabilities and log-probabilities.
    log_q = F.log_softmax(log_alpha, dim=-1)     # shape: [B, N, C]
    q = log_q.exp()                              # shape: [B, N, C]

    # Compute batch aggregated marginals (per latent dimension).
    batch_q_marginals = q.mean(dim=0)            # shape: [N, C]

    # Update running aggregated marginals using EMA.
    if running_q_marginals is None:
        running_q_marginals = batch_q_marginals.detach()
    else:
        running_q_marginals = momentum * running_q_marginals + (1 - momentum) * batch_q_marginals.detach()

    # Compute log of running aggregated marginals.
    log_q_marginals = torch.log(running_q_marginals + eps)  # shape: [N, C]

    # Compute per-sample Mutual Information.
    # For each sample b and latent j:
    #   MI[b,j] = sum_i q[b, j, i] * (log_q[b, j, i] - log_q_marginals[j, i])
    mi_per_latent = (q * (log_q - log_q_marginals.unsqueeze(0))).sum(dim=-1)  # shape: [B, N]
    mi_per_sample = mi_per_latent.sum(dim=-1)  # shape: [B]

    # Compute per-sample overall KL divergence.
    # For each sample b and latent j:
    #   KL[b,j] = sum_i q[b, j, i] * (log_q[b, j, i] - log p(i))
    # For a uniform prior, log p(i) = -log(C)
    prior_log = -math.log(C)
    kl_per_latent = (q * (log_q - prior_log)).sum(dim=-1)  # shape: [B, N]
    kl_per_sample = kl_per_latent.sum(dim=-1)              # shape: [B]

    # Compute Dimension-Wise KL (global) using the running aggregated marginals.
    # DWKL = sum_{j=1}^{N} sum_{i=1}^{C} running_q_marginals[j, i] * (log(running_q_marginals[j, i] + eps) - (-log(C)))
    #      = sum_{j=1}^{N} sum_{i=1}^{C} running_q_marginals[j, i] * (log(running_q_marginals[j, i] + eps) + log(C))
    dw_kl = (running_q_marginals * (torch.log(running_q_marginals + eps) + math.log(C))).sum()  # scalar

    # Compute per-sample Total Correlation.
    # TC[b] = KL[b] - MI[b] - DWKL (DWKL is global, so subtracted from each sample)
    tc_per_sample = kl_per_sample - mi_per_sample - dw_kl  # shape: [B]

    # Apply reduction.
    if reduction == "sum":
        mutual_info_reduced = mi_per_sample.sum()
        overall_kl_reduced = kl_per_sample.sum()
        total_corr_reduced = tc_per_sample.sum()
    elif reduction in ["mean", "batchmean"]:
        scale = 1/N if reduction == "mean" else 1
        # Optionally, if you want the mean per latent, divide by N.
        mutual_info_reduced = mi_per_sample.mean() * scale
        overall_kl_reduced = kl_per_sample.mean() * scale
        total_corr_reduced = tc_per_sample.mean() * scale
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return KLDLosses(
        mutual_info=mutual_info_reduced,
        total_correlation=total_corr_reduced,
        dimension_wise_kl=dw_kl,
        overall_kl=overall_kl_reduced
    ), running_q_marginals


def approximate_kld_loss(log_alpha, eps=1e-6, reduction='sum'):
    """
    Compute KL divergence using Eric Jang's trick. 
    Here we completely disregard the sample (relaxed value using temperature) of the distribution
    and use the normalised logits to directly compute the KL divergence.
    
    Args:
        log_alpha: Logits of the distribution (unnormalised log alpha) [B, N, C]
        eps: Small constant for numerical stability
        reduction: Reduction method (sum, mean, batchmean)
    Returns:
        KL divergence loss
    """
    
    # [B, N, C]
    normalised_log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    B, N, C = normalised_log_alpha.shape
    # Compute KL divergence using Eric Jang's trick

    # No need to use softmax as we are exponentiating the normalised logits
    probs = torch.exp(normalised_log_alpha).clamp_min(eps)
    log_prior = torch.full_like(normalised_log_alpha, -torch.log(torch.tensor(C)))
    kld = (probs * (normalised_log_alpha - log_prior)).sum(dim=-1) # [B, N]

    if reduction == 'sum':
        return KLDLosses(overall_kl=kld.sum())
    elif reduction == 'mean':
        return KLDLosses(overall_kl=kld.mean())
    elif reduction == 'batchmean':
        return KLDLosses(overall_kl=kld.sum(1).mean(dim=0))
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def monte_carlo_kld(log_alpha, tau, reduction='sum', use_exp_relaxed=False):
    """
    Compute KL divergence using Monte Carlo estimation.
    In this case, we create distributions from the original logits, sample,
    and then use the probability density functions to obtain the sample estimate.

    If we want even better estimate, we can call this function multiple times and average the results.
    
    Args:
        log_alpha: Original logits from forward pass [B, N, C]
        tau: Temperature parameter
        
    Returns:
        KL divergence loss
    """

    #[B, N, C] -> [B, N] Batch Shape, [C] Event Shape

    codebook_size = log_alpha.shape[-1]
    tau = tau.clone().detach() if isinstance(tau, torch.Tensor) else torch.tensor(tau)
    # Create posterior distribution from original logits
    # It is important that the temperature use the same temperature as the one used in the forward pass
    # Even for the prior distribution
    if use_exp_relaxed:
        v_dist = ExpRelaxedCategorical(tau, logits=log_alpha)
        prior = ExpRelaxedCategorical(tau, probs=torch.ones(codebook_size, device=log_alpha.device))
    else:
        v_dist = RelaxedOneHotCategorical(tau, logits=log_alpha)
        prior = RelaxedOneHotCategorical(tau, probs=torch.ones(codebook_size, device=log_alpha.device))
    
    # Sample from the posterior distribution
    z = v_dist.rsample() # relaxed sample from the posterior distribution
    
    # Compute KL divergence using Monte Carlo estimation
    n_batch = log_alpha.shape[0]
    n_latent = log_alpha.shape[1]

    prior_expanded = prior.expand(torch.Size([n_batch, n_latent]))

    # [B, N, C] -> [B, N] (Batch Shape) because it collapsed on the event shape (even for relaxed sample z)
    kld = (v_dist.log_prob(z) - prior_expanded.log_prob(z)) # [B, N] (Batch Shape)
    
    if reduction == 'sum':
        return KLDLosses(overall_kl=kld.sum())
    elif reduction == 'mean':
        return KLDLosses(overall_kl=kld.mean())
    elif reduction == 'batchmean':
        return KLDLosses(overall_kl=kld.sum(1).mean(dim=0))
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
