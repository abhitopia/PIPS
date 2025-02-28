#!/usr/bin/env python3
import torch
import torch.nn.functional as F

def test_vanilla_softmax(logits, temperature, num_samples=5):
    """
    Computes vanilla softmax with temperature scaling and prints the output.
    
    Note: Vanilla softmax is deterministic. Temperature scaling is applied
    as softmax(logits/temperature).
    """
    print(f"\nVanilla Softmax (tau={temperature}):")
    # Vanilla softmax is deterministic so we calculate it once.
    soft = F.softmax(logits / temperature, dim=-1)
    for i in range(num_samples):
        print(f"Sample {i+1}: {soft.numpy()}")

def test_gumbel_softmax(logits, temperature, hard=False, num_samples=5):
    """
    Samples from the Gumbel softmax distribution multiple times and prints the output.
    
    Args:
        logits (Tensor): The input logits.
        temperature (float): Temperature parameter.
        hard (bool): Whether to return one-hot samples using the straight-through estimator.
        num_samples (int): Number of samples to generate.
    """
    print(f"\nGumbel Softmax samples (tau={temperature}, hard={hard}):")
    for i in range(num_samples):
        sample = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        print(f"Sample {i+1}: {sample.numpy()}")

def main():
    # For reproducibility
    torch.manual_seed(42)

    # Define a set of logits
    logits = torch.tensor([0.1, 0.2, 0.7])
    print("Logits:", logits.numpy())

    # Test vanilla softmax with temperature scaling.
    test_vanilla_softmax(logits, temperature=1.0)
    test_vanilla_softmax(logits, temperature=0.01)

    # Test Gumbel softmax with different settings.
    test_gumbel_softmax(logits, temperature=1.0, hard=False)
    test_gumbel_softmax(logits, temperature=0.01, hard=False)
    test_gumbel_softmax(logits, temperature=0.01, hard=True)

if __name__ == '__main__':
    main()
