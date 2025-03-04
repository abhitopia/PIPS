import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

class Codebook(nn.Module):
    def __init__(self, d_model, codebook_size):
        super().__init__()
        self.head = nn.Linear(d_model, codebook_size, bias=False)
        self.codebook = nn.Linear(codebook_size, d_model, bias=False)

    def sample(self, soft_code, logits, reinMax: bool = False):
          # Straight through
        index = soft_code.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(
            soft_code, memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, index, 1.0)
        
        if reinMax:
            # ReinMax: Algorithm 2 in https://arxiv.org/pdf/2304.08612
            # Notice that I use pi_0 from gumbel instead of the softmax, this is deliberate
            # The noise in gumbel softmax better captures the stochasticity of sampling
            # For example, even in STE (Algorithm 1), the authors don't use gumbel softmax as pi_0
            # However, even the official gumbel softmax implementation in PyTorch uses the softmax with gumbel noise
            pi_0 = soft_code # Step 1
            D = y_hard # Step 2
            pi_1 = (D + pi_0) / 2
            pi_1 = F.softmax((pi_1.log() - logits).detach() - logits, dim=-1)
            pi_2 = 2 * pi_1 - 0.5 * pi_0
            hard_code = pi_2 - pi_2.detach() + D
        else:
            hard_code = y_hard - soft_code.detach() + soft_code

        return hard_code

    def forward(self, logits, tau: float = 1.0, hardness: Tensor = torch.tensor(0.0), reinMax: bool = False):
        logits = self.head(logits)

        # Compute both versions of soft_code.
        soft_code_softmax = F.softmax(logits / tau, dim=-1)
        soft_code_gumbel = F.gumbel_softmax(logits, tau=tau, hard=False)
        
        # Use hardness to select between softmax and gumbel-softmax.
        # (Assuming hardness < 0 means use softmax, otherwise use gumbel-softmax)
        cond_soft = hardness < 0
        # Expand the condition to match the shape of the soft code if needed.
        cond_soft = cond_soft.expand_as(soft_code_softmax) if soft_code_softmax.dim() > 0 else cond_soft
        soft_code = torch.where(cond_soft, soft_code_softmax, soft_code_gumbel)
        
        # Compute the hard code branch.
        hard_code = self.sample(soft_code, logits=logits, reinMax=reinMax)
        
        # If hardness > 0, use a linear combination of the hard and soft codes.
        # Otherwise, just use soft_code.
        cond_code = hardness > 0
        combined = hardness * hard_code + (1 - hardness) * soft_code
        # Again, expand the condition if necessary.
        cond_code = cond_code.expand_as(soft_code) if soft_code.dim() > 0 else cond_code
        code = torch.where(cond_code, combined, soft_code)
    
        return self.codebook(code), soft_code, code

    def forward_no_compile(self, logits, tau: float = 1.0, hardness: float = 0.0, reinMax: bool = False):
        logits = self.head(logits)
        
        if hardness < 0:
            # Compute both softmax and gumbel-softmax versions
            soft_code = F.softmax(logits/tau, dim=-1)
        else:
            soft_code = F.gumbel_softmax(logits, tau=tau, hard=False)
            
        # For the hard code path, also use tensor operations
        if hardness > 0:
            hard_code = self.sample(soft_code, logits=logits, reinMax=reinMax)
            code = hardness * hard_code + (1 - hardness) * soft_code
        else:
            code = soft_code

        return self.codebook(code), soft_code, code

    @staticmethod
    def kld_disentanglement_loss(q_z_x, q_z_marg=None, momentum=0.99, eps=1e-8):
        """
        The Beta-TCVAE paper(https://arxiv.org/pdf/1802.04942) splits the KLD term as

        1/N KLD(q(z|x) | p(z)) = MI + TC + DWKL

        where, 
            MI = KLD(q(z,x)|q(z)p(x))
            TC = KLD(q(z)|∏[j] q(z_j))
            DWKL = sum_j KLD(q(z_j) | p(z_i))

        Computing q(z, x), q(z) and q(z_i) are all intractable. This happens because the
        authors compute average across all the dataset which involves p(x_n) and makes life
        complicated. I suspect, this is useful for when the dataset size is small and fixed.
        However, I want to operate in the large/infinite data regime. So I will try to compute
        this first per sample x_n (like it done for VAE) and average for the batch later.

        Using similar derivation as in the paper, we get following

        KL_n(q(z| x_n) | p(z)) = MI_n + TC_n + DWKL_n
        where 
            MI_n   = E{q(z|x_n)}[log q(z|x_n)/q(z)]
            TC_n   = E{q(z|x_n)}[log q(z)/ ∏{j} q(z_j)]
            DWKL_n = Sum_j E{q(z|x_n)}[log q(z_j)/p(z_j)]

        Notice that since, expectation is a sum over all latents, this will be C^N
        different values making it intractable. So regular VAE (computing LHS directly),
        we have to assume assume:
        - The latent vector z = (z_1, z_2, ..., z_N) factorizes:
                q(z|x) = ∏[j=1 to N] q(z_j|x)
        - The prior also factorizes:
                p(z) = ∏[j=1 to N] p(z_j)

        The encoder gives us access to q(z|x_n), but we don't have access to aggregate posterior
        q(z), or it's marginal per latent q(z_i) which are global quantities (not dependent on the batch).
        The original paper uses MWS and MSS sampling to estimate these which are too complicated and ineffecient. 

        We can use Monte Carlo sampling to estimate q(z_i) (And not q(z)) as following

        q(z_i) = sum_n q(z_i |x_n) p(x_n)  # Per mini-batch of size B
               = 1/ B sum_n q(z_i | x_n)   # (N, C)          

        Because a batch is not a good representation of p(x), we can use EMA to update this over time.

        But both MI_n and TC_n still involve q(z) which remains intractable. We cannot assume that q(z)
        factorises to prod_j q(z), because if we did, then TC_n would vanish. In fact, this is the term
        that encourages disentanglement. What should we do?

        My idea here is that we approximate MI_n with the assumption that q(z) factorises, and then use
        the following identity to estimate TC_n

        TC_n = KL_n - MI_n (approx) - DWKL_n

        where 
        
        MI_n (approx) = E{q(z|x_n)}[log q(z|x_n)/∏{j} q(z_j)]
                      = E{q(z|x_n)}[log ∏{j} (q(z_j|x_n)/ q(z_j)]
                      = sum_j E{q(z|x_n)} log(q(z_j|x_n) / q(z_j))
                      = sum_j E{q(z_j|x_n)} log(q(z_j|x_n) / q(z_j))

        Compute disentanglement losses—Full KL, Mutual Information (MI), 
        Dimension-wise KL (DWKL), and Total Correlation (TC)—using an exponential moving 
        average (EMA) to estimate the aggregated posterior q(z).

        This function assumes:
        - The latent vector z = (z_1, z_2, ..., z_N) factorizes:
                q(z|x) = ∏[j=1 to N] q(z_j|x)
        - The prior also factorizes:
                p(z) = ∏[j=1 to N] p(z_j)

        For specific x, therefore, the full KL (per sample) is defined as:
            KL(q(z|x) || p(z)) = Σ[j=1 to N] KL(q(z_j|x) || p(z_j))

        The paper provides a decomposition of the "average KL" across all samples in the dataset
        which makes life complicated because we don't have access to p(x_n). Instead, following is the 
        rederivation of the decomposition (like usually done for VAEs) for each specific sample x_i.

        KL(q(z|x_i) || p(z)) = KL(q(z|x_i) || p(z))

        = E{q(z|x_i)} [log q(z|x_i) - log p(z)]

        = E{q(z|x_i)} [log q(z|x_i) - log ∏ p(z_j) + log q(z) - log q(z) + log ∏ q(z_j) - log ∏ q(z_j) ]

        = E{q(z|x_i)} [log q(z|x_i)/q(z) + log q(z)/∏ q(z_j) + log ∏ q(z_j)/p(z_j) ]

        = Σ{z} [q(z|x_i) * (log q(z|x_i)/log q(z) + log q(z)/∏ q(z_j) + Σ{j} log q(z_j)/p(z_j) )]

        = Σ{z} [q(z|x_i) * (log q(z|x_i)/log q(z) + log q(z)/∏ q(z_j) + Σ{z_j} log q(z_j)/p(z_j) )]


        MI = Σ{z} [q(z|x_i) * log q(z|x_i)/log q(z)]
        TC = Σ{z} [q(z|x_i) * (log q(z) - Σ{j} log q(z_j)]
        DWKL = Σ{z} [q(z|x_i) * Σ{z_j} log q(z_j)/p(z_j)]
      

        Also, following the decomposition:
            Full KL = MI + TC + DWKL,
        where:
            - MI = Σ[j=1 to N] KL(q(z_j|x) || q(z_j))  (averaged over samples),
            - DWKL = Σ[j=1 to N] KL(q(z_j) || p(z_j))    (computed from an estimate of q(z), 
                                                        ideally a global quantity),
            - TC = Full KL - MI - DWKL.

        To ensure consistency, we compute Full KL and MI per sample (summing over latent dimensions, then averaging over the batch) 
        and compute DWKL from the aggregated marginal. Because a minibatch may not reliably capture the global posterior, 
        we use an EMA to keep a running estimate of q(z).

        Args:
            q_z_x (Tensor): Soft one-hot distributions from Gumbel-softmax, of shape (B, N, C), where:
                B = batch size,
                N = number of discrete latent codes,
                C = codebook size.
            q_z_marg: Optional tensor of shape (N, C) containing the running estimate of q(z)
            momentum (float): The decay rate for the EMA; typical values are near 0.99
            eps (float): A small constant for numerical stability
            apply_relu (bool): Whether to apply ReLU to ensure non-negative losses

        Returns:
            Tuple containing:
            - Dictionary of KL-related losses
            - Updated marginal q(z)
        """
        # -----------------------------------------------
        # Retrieve dimensions.
        # B: batch size, N: number of latent codes, C: codebook size.
        # -----------------------------------------------
        B, N, C = q_z_x.shape

        # -----------------------------------------------
        # Step 1: Compute the current aggregated marginal posterior q(z_j) from the minibatch.
        # For each latent dimension j, compute q_z_current[j] as the average over the batch.
        # q_z_marg_current has shape (N, C). Each row is a valid probability distribution (sums to 1).
        # -----------------------------------------------
        q_z_marg_batch = q_z_x.mean(dim=0)  # Shape: (N, C)

        # -----------------------------------------------
        # Step 1b: Update the running estimate of q(z_j) using EMA.
        # If q_z_marg_running is provided, update it; otherwise, initialize it with q_z_marg_current.
        # The formula is: q_z_running_new = momentum * q_z_running_old + (1 - momentum) * q_z_current
        # This provides a smoother estimate of the global q(z) than using the current batch alone.
        # -----------------------------------------------
        q_z_marginal = (momentum * q_z_marg + (1 - momentum) * q_z_marg_batch) if q_z_marg is not None else q_z_marg_batch
        q_z_marginal_detached = q_z_marginal.detach()  # Detach for return value

        # Compute log probabilities
        log_q_z_x = torch.log(q_z_x + eps)  # Shape: (B, N, C)
        # Since p(z_j) is assumed to be uniform over C possible codes, each entry is 1/C.
        log_uniform = torch.full_like(log_q_z_x, -math.log(C))  # Shape: (1, N, C)

        # -----------------------------------------------
        # Step 2: Compute the Full KL Divergence per sample.
        #
        # For each sample i and latent dimension j, compute:
        #   KL(q(z_j|x_i) || p(z_j)) = Σ[c=1 to C] q(z_j=c|x_i) * 
        #                                ( log(q(z_j=c|x_i)) - log(1/C) )
        #
        # This gives a tensor of shape (B, N) (KL per latent, per sample).
        # Then, sum over the latent dimensions (for each sample) to get the full KL per sample.
        # Finally, average over the batch to obtain the final full KL loss.
        # -----------------------------------------------
        kl_per_latent = (q_z_x * (log_q_z_x - log_uniform)).sum(dim=-1)  # Shape: (B, N)
        full_kl_batch = kl_per_latent.sum(dim=1)  # Sum over latent dimensions → Shape: (B,)


        # -----------------------------------------------
        # Step 3: Compute Dimension-wise KL (DWKL) from aggregated posteriors.
        #   DWKL_n = Σ[j=1 to N] E{q(z|x_n)}[log q(z_j)/p(z_j)]
        #          = Σ[j=1 to N] E{q(z_j |x_n)}[log q(z_j)/p(z_j)]   (others sum to 1)
        log_q_z_marg = torch.log(q_z_marginal + eps).unsqueeze(0)  # Shape: (1, N, C)

        # For each latent dimension j, using the running aggregated posterior marginal q_z_running,
        # compute:
        #   DWKL(j) = KL(q(z_j) || p(z_j)) = Σ[c=1 to C] q_z_x[j, c] *
        #                                   ( log(q_z[j, c]) - log(1/C) )
        #
        # This yields a tensor of shape (N,), one value per latent dimension.
        # Sum over all latent dimensions to yield the total DWKL.
        # -----------------------------------------------
        dwkl_per_latent  = (q_z_x * (log_q_z_marg - log_uniform)).sum(dim=-1)  # Shape: (B, N)
        dwkl_batch = dwkl_per_latent.sum(dim=-1)  #  (B,) 

        # -----------------------------------------------
        # Step 3: Compute Mutual Information (MI)
        # NOTE: As explained above, assume here that q(z) factorises into q(z_j)
        log_q_z = torch.log(q_z_marginal + eps)  # Shape: (N, C) (Assume q(z) factorises)

        # For each sample i and latent dimension j, compute:
        #   MI_component = KL(q(z_j|x_i) || q(z_j)) = Σ[c=1 to C] q(z_j=c|x_i) * 
        #                     ( log(q(z_j=c|x_i)) - log(q_z_running[j, c]) )
        #
        # q_z_running is the EMA estimate for the aggregated posterior.
        # This gives a tensor of shape (B, N).
        # Sum over latent dimensions for each sample, then average over the batch.
        # -----------------------------------------------
        mi_per_latent = (q_z_x * (log_q_z_x - log_q_z.unsqueeze(0))).sum(dim=-1)  # Shape: (B, N)
        mi_batch = mi_per_latent.sum(dim=1)  # Sum over latent → Shape: (B,)


        # -----------------------------------------------
        # Step 5: Compute Total Correlation (TC) loss.
        # Estimating TC directly is challenging because it requires the joint distribution of all latent codes.
        # But we cannot also use the approximation q(z) = prod_j q(z_j) as otherwise TC will vanish
        # We can solve for TC:
        #   TC_n = KL_n - MI_n - DWKL_n
        # -----------------------------------------------
        tc_batch = full_kl_batch - mi_batch - dwkl_batch  # (B,) for each sample
       
        # -----------------------------------------------
        # Return the computed losses (averaged per sample over batch) and the updated running aggregated posterior.
        # -----------------------------------------------


        return {
            "mi_loss": mi_batch.mean(), 
            "dwkl_loss": dwkl_batch.mean(), 
            "tc_loss": tc_batch.mean(), 
            "kl_loss": full_kl_batch.mean()
        }, q_z_marginal_detached

