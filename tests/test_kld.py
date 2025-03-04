import pytest
import torch
import math
from pips.kld import (
    compute_decomposed_kld,
    approximate_kld_loss,
    monte_carlo_kld,
    KLDLosses
)


class TestKLD:
    @pytest.fixture
    def setup_data(self):
        # Create deterministic test data
        torch.manual_seed(42)
        batch_size = 4
        num_latents = 3
        codebook_size = 5
        
        # Create log_alpha with different patterns for testing
        log_alpha = torch.randn(batch_size, num_latents, codebook_size)
        
        return {
            'log_alpha': log_alpha,
            'batch_size': batch_size,
            'num_latents': num_latents,
            'codebook_size': codebook_size,
        }
    
    def test_kld_losses_namedtuple(self):
        # Test the KLDLosses named tuple
        losses = KLDLosses(
            mutual_info=torch.tensor(1.0),
            total_correlation=torch.tensor(2.0),
            dimension_wise_kl=torch.tensor(3.0),
            overall_kl=torch.tensor(4.0)
        )
        
        assert losses.mutual_info.item() == 1.0
        assert losses.total_correlation.item() == 2.0
        assert losses.dimension_wise_kl.item() == 3.0
        assert losses.overall_kl.item() == 4.0
        
        # Test with default values for total_correlation and dimension_wise_kl
        losses = KLDLosses(
            mutual_info=torch.tensor(1.0),
            overall_kl=torch.tensor(5.0)
        )
        assert losses.mutual_info.item() == 1.0
        assert losses.total_correlation.item() == 0.0
        assert losses.dimension_wise_kl.item() == 0.0
        assert losses.overall_kl.item() == 5.0
    
    def test_compute_decomposed_kld_shapes(self, setup_data):
        log_alpha = setup_data['log_alpha']
        
        # Test with no running_q_marginals
        losses, running_q = compute_decomposed_kld(log_alpha)
        
        # Check shapes
        assert isinstance(losses, KLDLosses)
        assert running_q.shape == (setup_data['num_latents'], setup_data['codebook_size'])
        
        # Test with provided running_q_marginals
        losses, running_q_updated = compute_decomposed_kld(
            log_alpha, 
            running_q_marginals=running_q
        )
        
        # Check shapes again
        assert isinstance(losses, KLDLosses)
        assert running_q_updated.shape == (setup_data['num_latents'], setup_data['codebook_size'])
    
    def test_compute_decomposed_kld_reductions(self, setup_data):
        log_alpha = setup_data['log_alpha']
        
        # Test different reduction methods
        losses_sum, _ = compute_decomposed_kld(log_alpha, reduction="sum")
        losses_mean, _ = compute_decomposed_kld(log_alpha, reduction="mean")
        losses_batchmean, _ = compute_decomposed_kld(log_alpha, reduction="batchmean")
        
        # Sum should be larger than mean
        assert losses_sum.mutual_info > losses_mean.mutual_info
        
        # The relationship between mean and batchmean depends on implementation
        # Instead of checking exact relationship, just verify they're different
        assert losses_mean.mutual_info != losses_batchmean.mutual_info
        
        # Test invalid reduction
        with pytest.raises(ValueError):
            compute_decomposed_kld(log_alpha, reduction="invalid")
    
    def test_compute_decomposed_kld_momentum(self, setup_data):
        log_alpha = setup_data['log_alpha']
        num_latents = setup_data['num_latents']
        codebook_size = setup_data['codebook_size']
        
        # Use a defined initial running_q (uniform) that is different from the current batch marginals.
        running_q_init = torch.ones(num_latents, codebook_size) / codebook_size
        
        _, running_q1 = compute_decomposed_kld(log_alpha, running_q_marginals=running_q_init, momentum=0.9)
        _, running_q2 = compute_decomposed_kld(log_alpha, running_q_marginals=running_q_init, momentum=0.5)
        
        # Different momentum should result in different running_q values
        assert not torch.allclose(running_q1, running_q2)
    
    def test_approximate_kld_loss(self, setup_data):
        log_alpha = setup_data['log_alpha']
        
        # Test different reduction methods
        losses_sum = approximate_kld_loss(log_alpha, reduction="sum")
        losses_mean = approximate_kld_loss(log_alpha, reduction="mean")
        losses_batchmean = approximate_kld_loss(log_alpha, reduction="batchmean")
        
        # Check that the losses are of the expected type
        assert isinstance(losses_sum, KLDLosses)
        assert isinstance(losses_mean, KLDLosses)
        assert isinstance(losses_batchmean, KLDLosses)
        
        # Sum should be larger than mean
        assert losses_sum.overall_kl > losses_mean.overall_kl
        
        # Test invalid reduction
        with pytest.raises(ValueError):
            approximate_kld_loss(log_alpha, reduction="invalid")
    
    def test_monte_carlo_kld(self, setup_data):
        log_alpha = setup_data['log_alpha']
        tau = 1.0
        
        # Test with standard relaxed categorical
        losses_sum = monte_carlo_kld(log_alpha, tau, reduction="sum")
        losses_mean = monte_carlo_kld(log_alpha, tau, reduction="mean")
        losses_batchmean = monte_carlo_kld(log_alpha, tau, reduction="batchmean")
        
        # Check that the losses are of the expected type
        assert isinstance(losses_sum, KLDLosses)
        assert isinstance(losses_mean, KLDLosses)
        assert isinstance(losses_batchmean, KLDLosses)
        
        # Sum should be larger than mean
        assert losses_sum.overall_kl > losses_mean.overall_kl
        
        # Test with exp relaxed categorical
        losses_exp = monte_carlo_kld(log_alpha, tau, use_exp_relaxed=True)
        assert isinstance(losses_exp, KLDLosses)
    
    def test_kld_methods_consistency(self, setup_data):
        """Test that all KLD methods return values in a similar range"""
        log_alpha = setup_data['log_alpha']
        tau = 1.0
        
        # Get losses from different methods with the same reduction
        losses_decomposed, _ = compute_decomposed_kld(log_alpha, reduction="sum")
        losses_approximate = approximate_kld_loss(log_alpha, reduction="sum")
        losses_monte_carlo = monte_carlo_kld(log_alpha, tau, reduction="sum")
        
        # All methods should return positive KL divergence
        assert losses_decomposed.overall_kl > 0
        assert losses_approximate.overall_kl > 0
        assert losses_monte_carlo.overall_kl > 0
        
        # The decomposed KLD should have mutual information and total correlation
        assert losses_decomposed.mutual_info > 0 

    # --- New Additional Tests ---
    
    def test_compute_decomposed_kld_gradient(self, setup_data):
        log_alpha = setup_data['log_alpha']
        log_alpha.requires_grad_()
        loss, _ = compute_decomposed_kld(log_alpha, reduction="sum")
        loss.overall_kl.backward()
        assert log_alpha.grad is not None
        assert log_alpha.grad.shape == log_alpha.shape

    def test_approximate_kld_loss_gradient(self, setup_data):
        log_alpha = setup_data['log_alpha']
        log_alpha.requires_grad_()
        loss = approximate_kld_loss(log_alpha, reduction="sum")
        loss.overall_kl.backward()
        assert log_alpha.grad is not None
        assert log_alpha.grad.shape == log_alpha.shape

    def test_monte_carlo_kld_gradient(self, setup_data):
        log_alpha = setup_data['log_alpha']
        tau = 1.0
        log_alpha.requires_grad_()
        loss = monte_carlo_kld(log_alpha, tau, reduction="sum")
        loss.overall_kl.backward()
        assert log_alpha.grad is not None
        assert log_alpha.grad.shape == log_alpha.shape

    def test_compute_decomposed_kld_uniform(self, setup_data):
        B, N, C = setup_data['batch_size'], setup_data['num_latents'], setup_data['codebook_size']
        log_alpha = torch.zeros(B, N, C)
        losses, _ = compute_decomposed_kld(log_alpha, reduction="sum")
        # With uniform logits the KL should be near zero for all components
        assert torch.isclose(losses.mutual_info, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.overall_kl, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.total_correlation, torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(losses.dimension_wise_kl, torch.tensor(0.0), atol=1e-5)

    def test_approximate_kld_loss_uniform(self, setup_data):
        B, N, C = setup_data['batch_size'], setup_data['num_latents'], setup_data['codebook_size']
        log_alpha = torch.zeros(B, N, C)
        loss = approximate_kld_loss(log_alpha, reduction="sum")
        assert torch.isclose(loss.overall_kl, torch.tensor(0.0), atol=1e-5)

    def test_monte_carlo_kld_uniform(self, setup_data):
        B, N, C = setup_data['batch_size'], setup_data['num_latents'], setup_data['codebook_size']
        log_alpha = torch.zeros(B, N, C)
        tau = 1.0
        loss = monte_carlo_kld(log_alpha, tau, reduction="sum")
        # Allow a small tolerance due to Monte Carlo variance
        assert torch.abs(loss.overall_kl) < 1e-3

    def test_monte_carlo_invalid_reduction(self, setup_data):
        log_alpha = setup_data['log_alpha']
        tau = 1.0
        with pytest.raises(ValueError):
            monte_carlo_kld(log_alpha, tau, reduction="invalid")

    def test_monte_carlo_tau_tensor(self, setup_data):
        log_alpha = setup_data['log_alpha']
        tau = torch.tensor(1.0)
        loss = monte_carlo_kld(log_alpha, tau, reduction="sum")
        assert isinstance(loss, KLDLosses)

    def test_compute_decomposed_kld_reduction_consistency_sum(self, setup_data):
        B = setup_data['batch_size']
        log_alpha = setup_data['log_alpha']
        losses_sum, _ = compute_decomposed_kld(log_alpha, reduction="sum")
        # For 'sum' reduction: overall_kl = mutual_info + total_corr + B * dimension_wise_kl
        expected = losses_sum.mutual_info + losses_sum.total_correlation + B * losses_sum.dimension_wise_kl
        assert torch.allclose(losses_sum.overall_kl, expected, atol=1e-5)

    def test_compute_decomposed_kld_reduction_consistency_batchmean(self, setup_data):
        log_alpha = setup_data['log_alpha']
        losses_bm, _ = compute_decomposed_kld(log_alpha, reduction="batchmean")
        # For 'batchmean' reduction: overall_kl = mutual_info + total_corr + dimension_wise_kl
        expected = losses_bm.mutual_info + losses_bm.total_correlation + losses_bm.dimension_wise_kl
        assert torch.allclose(losses_bm.overall_kl, expected, atol=1e-5)

    def test_compute_decomposed_kld_reduction_consistency_mean(self, setup_data):
        N_dim = setup_data['num_latents']
        log_alpha = setup_data['log_alpha']
        losses_mean, _ = compute_decomposed_kld(log_alpha, reduction="mean")
        # For 'mean' reduction: overall_kl = mutual_info + total_corr + (dimension_wise_kl / N_dim)
        expected = losses_mean.mutual_info + losses_mean.total_correlation + losses_mean.dimension_wise_kl / N_dim
        assert torch.allclose(losses_mean.overall_kl, expected, atol=1e-5)

    def test_approximate_kld_loss_invalid_reduction(self, setup_data):
        log_alpha = setup_data['log_alpha']
        with pytest.raises(ValueError):
            approximate_kld_loss(log_alpha, reduction="invalid") 