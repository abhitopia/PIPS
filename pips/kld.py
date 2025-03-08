from dataclasses import dataclass, field
import math
import torch
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical, RelaxedOneHotCategorical
from torch.nn import functional as F
from typing import Any

@dataclass
class KLDLosses:
    overall_kl: torch.Tensor
    mutual_info: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    total_correlation: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    dimension_wise_kl: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def to_dict(self) -> dict:
        return {
            "kl_loss": self.overall_kl,
            "mi_loss": self.mutual_info, 
            "tc_loss": self.total_correlation, 
            "dwkl_loss": self.dimension_wise_kl 
        }


def compute_decomposed_kld(
    log_alpha, 
    q_z_marginals=None, 
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
        q_z_marginals: Tensor of shape [N, C] representing the running aggregated marginals.
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
    # -----------------------------------------------
    # Retrieve dimensions.
    # B: batch size, N: number of latent codes, C: codebook size.
    # -----------------------------------------------
    B, N, C = log_alpha.shape

    # Compute soft probabilities and log-probabilities.
    log_q_x_z = F.log_softmax(log_alpha, dim=-1)     # shape: [B, N, C]
    q_z_x = log_q_x_z.exp()                              # shape: [B, N, C]



    # -----------------------------------------------
    # Step 1: Compute the current aggregated marginal posterior q(z_j) from the minibatch.
    # For each latent dimension j, compute q_z_current[j] as the average over the batch.
    # q_z_marg_current has shape (N, C). Each row is a valid probability distribution (sums to 1).
    # -----------------------------------------------
    # Compute batch aggregated marginals (per latent dimension).
    q_z_marg_batch = q_z_x.mean(dim=0)            # shape: [N, C] (q_z_marg_batch)


     # -----------------------------------------------
    # Step 1b: Update the running estimate of q(z_j) using EMA.
    # If q_z_marg_running is provided, update it; otherwise, initialize it with q_z_marg_current.
    # The formula is: q_z_running_new = momentum * q_z_running_old + (1 - momentum) * q_z_current
    # This provides a smoother estimate of the global q(z) than using the current batch alone.
    # -----------------------------------------------
    # Update running aggregated marginals using EMA.
    if q_z_marginals is None:
        q_z_marginals = q_z_marg_batch
    else:
        q_z_marginals = momentum * q_z_marginals + (1 - momentum) * q_z_marg_batch

    q_z_marginals_detached = q_z_marginals.detach()
    
    # -----------------------------------------------
    # Step 1: Compute the Full KL Divergence per sample.
    #
    # For each sample i and latent dimension j, compute:
    #   KL(q(z_j|x_i) || p(z_j)) = Σ[c=1 to C] q(z_j=c|x_i) * 
    #                                ( log(q(z_j=c|x_i)) - log(1/C) )
    #
    # This gives a tensor of shape (B, N) (KL per latent, per sample).
    # Then, sum over the latent dimensions (for each sample) to get the full KL per sample.
    # Finally, average over the batch to obtain the final full KL loss.
    # -----------------------------------------------
    prior_log = -math.log(C)
    kl_per_latent = (q_z_x * (log_q_x_z - prior_log)).sum(dim=-1)  # shape: [B, N]
    kl_per_sample = kl_per_latent.sum(dim=-1)              # shape: [B]


    # Compute log of running aggregated marginals.
    log_q_marginals = torch.log(q_z_marginals + eps)  # shape: [N, C]

    # -----------------------------------------------
    # Step 3: Compute Dimension-Wise KL (DWKL)
    # Compute Dimension-Wise KL (global) using the running aggregated marginals.
    # For each latent dimension j, using the running aggregated posterior marginal q_z_running,
    # compute:
    #   DWKL(j) = KL(q(z_j) || p(z_j)) = Σ[c=1 to C] q_z_x[j, c] *
    #                                   ( log(q_z[j, c]) - log(1/C) )
    #
    # This yields a tensor of shape (N,), one value per latent dimension.
    # Sum over all latent dimensions to yield the total DWKL.
    # -----------------------------------------------
    dwkl_per_latent = (q_z_x * (log_q_marginals.unsqueeze(0) - prior_log)).sum(dim=-1)  # shape: [B, N]
    dwkl_per_sample = dwkl_per_latent.sum(dim=-1)  # shape: [B]


    # -----------------------------------------------
    # Step 4: Compute Mutual Information (MI)
    # NOTE: As explained above, assume here that q(z) factorises into q(z_j)
    # For each sample i and latent dimension j, compute:
    #   MI_component = KL(q(z_j|x_i) || q(z_j)) = Σ[c=1 to C] q(z_j=c|x_i) * 
    #                     ( log(q(z_j=c|x_i)) - log(q_z_running[j, c]) )
    #
    # q_z_running is the EMA estimate for the aggregated posterior.
    # This gives a tensor of shape (B, N).
    # Sum over latent dimensions for each sample, then average over the batch.
    # -----------------------------------------------
    mi_per_latent = (q_z_x * (log_q_x_z - log_q_marginals.unsqueeze(0))).sum(dim=-1)  # shape: [B, N]
    mi_per_sample = mi_per_latent.sum(dim=-1)  # shape: [B]


    # -----------------------------------------------
    # Step 5: Compute Total Correlation (TC) loss.
    # Estimating TC directly is challenging because it requires the joint distribution of all latent codes.
    # But we cannot also use the approximation q(z) = prod_j q(z_j) as otherwise TC will vanish
    # We can solve for TC:
    #   TC_n = KL_n - MI_n - DWKL_n
    # -----------------------------------------------
    tc_per_sample = kl_per_sample - mi_per_sample - dwkl_per_sample  # shape: [B]

    # Apply reduction.
    if reduction == "sum":
        mutual_info_reduced = mi_per_sample.sum()
        overall_kl_reduced = kl_per_sample.sum()
        total_corr_reduced = tc_per_sample.sum()
        dimension_wise_kl_reduced = dwkl_per_sample.sum()
    elif reduction in ["mean", "batchmean"]:
        scale = 1/N if reduction == "mean" else 1
        # Optionally, if you want the mean per latent, divide by N.
        mutual_info_reduced = mi_per_sample.mean() * scale
        overall_kl_reduced = kl_per_sample.mean() * scale
        total_corr_reduced = tc_per_sample.mean() * scale
        dimension_wise_kl_reduced = dwkl_per_sample.mean() * scale
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return KLDLosses(
        mutual_info=mutual_info_reduced,
        total_correlation=total_corr_reduced,
        dimension_wise_kl=dimension_wise_kl_reduced,
        overall_kl=overall_kl_reduced
    ), q_z_marginals_detached


def approximate_kld_loss(log_alpha: torch.Tensor, eps: float = 1e-6, reduction: str = 'sum') -> KLDLosses:
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


def monte_carlo_kld(log_alpha: torch.Tensor, tau: torch.Tensor, reduction: str = 'sum', use_exp_relaxed: bool = False) -> KLDLosses:
    """
    Compute KL divergence using Monte Carlo estimation.
    In this case, we create distributions from the original logits, sample,
    and then use the probability density functions to obtain the sample estimate.

    If we want an even better estimate, we can call this function multiple times and average the results.
    
    Args:
        log_alpha (Tensor): Original logits from forward pass [B, N, C]
        tau (Tensor): Temperature parameter as a scalar tensor.
        reduction (str, optional): Reduction method, one of 'sum', 'mean', or 'batchmean'. Default is 'sum'.
        use_exp_relaxed (bool, optional): Whether to use ExpRelaxedCategorical. Default is False.
        
    Returns:
        KLDLosses: A dataclass containing the KL divergence losses.
    """
    # Ensure tau is a valid positive scalar tensor.
    assert tau > 0.0, f"Temperature must be greater than 0.0, got {tau.item()}"

    # Compute the uniform logit.
    codebook_size = log_alpha.shape[-1]
    # Using -log(codebook_size) explicitly signals that each category has probability 1/C after softmax.
    prior_val = -torch.log(torch.tensor(codebook_size, device=log_alpha.device, dtype=log_alpha.dtype))
    # Create uniform logits for the prior with the same shape as log_alpha.
    prior_logits = torch.full_like(log_alpha, prior_val)

    # Create the posterior and prior distributions using the given logits.
    if use_exp_relaxed:
        v_dist = ExpRelaxedCategorical(tau, logits=log_alpha)
        prior = ExpRelaxedCategorical(tau, logits=prior_logits)
    else:
        v_dist = RelaxedOneHotCategorical(tau, logits=log_alpha)
        prior = RelaxedOneHotCategorical(tau, logits=prior_logits)
    
    # Sample from the posterior distribution.
    z = v_dist.rsample()  # relaxed sample from the posterior distribution
    
    # Compute KL divergence using Monte Carlo estimation
    # [B, N, C] -> [B, N] (Batch Shape) because it collapsed on the event shape (even for relaxed sample z)
    kld = (v_dist.log_prob(z) - prior.log_prob(z)) # [B, N] (Batch Shape)
    
    if reduction == 'sum':
        return KLDLosses(overall_kl=kld.sum())
    elif reduction == 'mean':
        return KLDLosses(overall_kl=kld.mean())
    elif reduction == 'batchmean':
        return KLDLosses(overall_kl=kld.sum(1).mean(dim=0))
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
