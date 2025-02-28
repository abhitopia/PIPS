import torch
import torch.nn.functional as F
import math

def compute_entropy(probs, eps=1e-12):
    """
    Computes the entropy (in nats) of a probability distribution.
    probs: Tensor of shape (batch, num_classes)
    """
    return -(probs * torch.log(probs.clamp(min=eps))).sum(dim=-1)

def sample_gumbel(shape, eps=1e-12):
    """
    Samples Gumbel noise of a given shape.
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def experiment_gumbel_softmax(logits, tau=1.0, noise_scale=1.0, num_samples=1000):
    """
    Applies scaled Gumbel noise to logits and computes the average entropy of the softmax output.
    
    logits: Tensor of shape (batch, num_classes)
    tau: Temperature parameter for softmax.
    noise_scale: A multiplier to scale the Gumbel noise.
    num_samples: How many times to sample and average the entropy.
    """
    entropies = []
    batch, num_classes = logits.shape
    for _ in range(num_samples):
        # Generate Gumbel noise with the same shape as logits
        gumbel_noise = sample_gumbel(logits.shape)
        # Perturb logits by adding scaled Gumbel noise, then divide by temperature
        perturbed_logits = logits + noise_scale * gumbel_noise
        probs = F.softmax(perturbed_logits / tau, dim=-1)
        # Compute average entropy over the batch
        ent = compute_entropy(probs).mean().item()
        entropies.append(ent)
    return sum(entropies) / len(entropies)

if __name__ == '__main__':
    torch.manual_seed(42)
    
    num_classes = 512
    batch_size = 8

    # Case 1: Zero logits (no bias, no learned signal)
    logits = torch.zeros(batch_size, num_classes)
    print("=== Zero Logits ===")
    for noise_scale in [1.0, 0.5, 0.2, 0.1]:
        avg_entropy = experiment_gumbel_softmax(logits, tau=1.0, noise_scale=noise_scale)
        print(f"Noise scale: {noise_scale:4.1f}, Temperature: 1.0 -> Avg Entropy: {avg_entropy:.4f} nats")
    
    # Case 2: Varying temperature with default noise_scale
    print("\n=== Varying Temperature ===")
    for tau in [1.0, 1.5, 2.0]:
        avg_entropy = experiment_gumbel_softmax(logits, tau=tau, noise_scale=1.0)
        print(f"Temperature: {tau:4.1f}, Noise scale: 1.0 -> Avg Entropy: {avg_entropy:.4f} nats")
    
    # Case 3: Nonzero logits via a bias spread
    print("\n=== Nonzero Logits (Bias Spread) ===")
    # Create a bias vector with a wide spread, e.g. uniformly from -1 to 1.
    bias = torch.empty(num_classes).uniform_(-1, 1)
    # Expand to (batch_size, num_classes)
    logits_bias = bias.unsqueeze(0).expand(batch_size, num_classes)
    avg_entropy = experiment_gumbel_softmax(logits_bias, tau=1.0, noise_scale=1.0)
    print(f"Nonzero logits with bias spread [-1,1]: Avg Entropy: {avg_entropy:.4f} nats")
