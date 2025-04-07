from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.amp import autocast


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTransposeBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAutoEncoder, self).__init__()

        # Ensure that out_channels / in_channels is a power of 2
        assert out_channels % in_channels == 0
        assert out_channels / in_channels == 2**int(math.log2(out_channels / in_channels))

        # Calculate the number of ConvBlocks needed
        num_blocks = int(math.log2(out_channels / in_channels))

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for i in range(num_blocks):
            in_ch_encoder = in_channels * 2**i
            out_ch_encoder = in_channels * 2**(i+1)

            in_ch_decoder = out_channels // 2**i
            out_ch_decoder = out_channels // 2**(i+1)

            self.encoder.append(ConvBlock(in_ch_encoder, out_ch_encoder))
            self.decoder.append(ConvTransposeBlock(in_ch_decoder, out_ch_decoder))


    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class TransformerAutoEncoder(nn.Module):
    def __init__(self, n_dim, n_layers, n_heads, seq_len, encode_norm=True):
        super(TransformerAutoEncoder, self).__init__()
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len

        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.n_dim)) if self.n_layers > 0 else torch.zeros(1, self.seq_len, self.n_dim, requires_grad=False)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.n_dim, nhead=self.n_heads, dim_feedforward=4*self.n_dim, batch_first=True, bias=False, norm_first=True),
            num_layers=self.n_layers,
            enable_nested_tensor=False
        ) if self.n_layers > 0 else nn.Identity()

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.n_dim, nhead=self.n_heads, dim_feedforward=4*self.n_dim, batch_first=True, bias=False, norm_first=True),
            num_layers=self.n_layers,
            enable_nested_tensor=False
        ) if self.n_layers > 0 else nn.Identity()

        self.encode_norm = nn.LayerNorm(self.n_dim) if encode_norm else nn.Identity()

    def encode(self, x):
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.encode_norm(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    

# Reference: https://github.com/Vrushank264/VQVAE-PyTorch/tree/main
class VectorQuantization(Function):
    """
    Custom autograd Function for vector quantization in a VQ-VAE.

    This function maps each input vector (of shape [B, N, C]) to the index of its closest
    embedding in the codebook (of shape [K, C]) using a squared L2 distance.
    
    Now, in addition to returning the indices, it also returns the corresponding minimum
    squared L2 distances (quantization errors) for each input vector.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens (or discrete codes)
      C = Embedding dimension
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Disable autocast to ensure full precision for distance calculations
        with autocast(device_type='cuda', enabled=False):
            # Cast to float32 explicitly for precise distance calculation
            inputs_fp32 = inputs.float()
            codebook_fp32 = codebook.float()
            
            # Expected shape: inputs is [B, N, C]
            B, N, C = inputs.shape

            # Flatten inputs to shape [B*N, C]
            flat_input = inputs_fp32.reshape(B * N, C)

            # Compute squared L2 norm of each codebook vector. Shape: [K]
            codebook_sq = torch.sum(codebook_fp32 * codebook_fp32, dim=1)

            # Compute squared L2 norm of each input vector. Shape: [B*N, 1]
            inputs_sq = torch.sum(flat_input * flat_input, dim=1, keepdim=True)

            # Compute squared Euclidean distance between each input and each codebook vector.
            # Using the identity: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*(x · e)
            # Resulting shape: [B*N, K]
            l2_dis = torch.addmm(input=codebook_sq + inputs_sq,
                                mat1=flat_input,
                                mat2=codebook_fp32.t(),
                                alpha=-2.0, beta=1.0)

            # For each input, find the index of the codebook vector with minimum distance.
            # Also retrieve the corresponding minimum distance.
            min_vals, idx_flat = torch.min(l2_dis, dim=1)

            # Reshape indices and distances back to shape [B, N]
            idx = idx_flat.reshape(B, N)
            distances = min_vals.reshape(B, N)

        # Mark these outputs as non-differentiable.
        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(distances)

        # Convert distances back to original dtype to maintain compatibility
        distances = distances.to(inputs.dtype)
        
        return idx, distances

    @staticmethod
    def backward(ctx, grad_outputs):
        raise RuntimeError("Backward pass is not defined for VectorQuantization. Use VQStraightThrough instead.")

VQ = VectorQuantization.apply


class VQStraightThrough(Function):
    """
    Custom autograd Function implementing the straight-through estimator for vector quantization.
    
    In the forward pass, it uses the updated VectorQuantization (VQ) to get the nearest codebook
    indices and the corresponding minimum distances (quantization errors). It then retrieves the 
    corresponding codebook embeddings (quantized vectors). In the backward pass, gradients are passed 
    directly (straight-through) to the encoder, while gradients for the codebook are accumulated 
    based on the quantization indices.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens
      C = Embedding dimension
    Codebook shape: [K, C]
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Since VQ now handles precision internally, we just need to ensure
        # that the subsequent operations also maintain appropriate precision
        
        # Get nearest codebook indices and corresponding quantization errors using updated VQ.
        idx, distances = VQ(inputs, codebook)  # This now runs in full precision
        
        # The following code should also be run in full precision
        with autocast(device_type='cuda', enabled=False):
            # Cast to float32 explicitly
            inputs_fp32 = inputs.float()
            codebook_fp32 = codebook.float()
            
            B, N, C = inputs.shape
            flat_idx = idx.reshape(B * N)
            
            # Save tensors for the backward pass
            ctx.save_for_backward(flat_idx, codebook_fp32)
            
            # Retrieve quantized embeddings via index selection
            codes_flat = torch.index_select(codebook_fp32, dim=0, index=flat_idx)
            codes = codes_flat.reshape(B, N, C)
            
            # Convert back to original dtype
            codes = codes.to(inputs.dtype)
        
        ctx.mark_non_differentiable(flat_idx)
        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(distances)
        
        return codes, flat_idx, idx, distances

    @staticmethod
    def backward(ctx, grad_outputs, grad_flat_idx, grad_idx, grad_distances):
        grad_inputs, grad_codebook = None, None
        
        # Pass gradients straight-through to the encoder.
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.clone()
        
        # Compute gradients with respect to the codebook.
        if ctx.needs_input_grad[1]:
            flat_idx, codebook = ctx.saved_tensors
            C = codebook.shape[1]
            flat_grad_output = grad_outputs.reshape(-1, C)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, flat_idx, flat_grad_output)
        
        return grad_inputs, grad_codebook

VQ_ST = VQStraightThrough.apply


class VQEmbedding(nn.Module):
    """
    VQEmbedding class for managing the codebook in a VQ-VAE when the encoder output
    comes from a transformer, i.e. with shape [B, N, C].

    This module creates a learnable embedding table (codebook) with K embeddings,
    each of dimension D. It provides two forward methods:
      - forward: Returns the discrete latent indices for a given encoded input.
      - straight_through_forward: Returns both the quantized vectors (using a
        straight-through estimator) and an alternative quantized representation
        obtained directly via index selection.

    Expected input shape:
      - z_e_x: (B, N, C), where:
            B = Batch size,
            N = Number of tokens (discrete codes),
            C = Embedding dimension (should equal D).
    """
    def __init__(self, codebook_size: int, n_dim: int, use_ema: bool = True, decay: float = 0.99,
                unused_reset_threshold: float = 1.0, 
                hot_reset_threshold: float = 500,
                distance_reset: bool = False):
        """
        Initialize the VQEmbedding module.
        
        Args:
            codebook_size (int): Total number of embeddings in the codebook.
            n_dim (int): Dimensionality of each embedding.
            decay (float): EMA decay rate (default: 0.99)
            unused_reset_threshold (float): Threshold below which a code is considered unused.
            hot_reset_threshold (float): Threshold above which a code is considered hot.
            distance_reset (bool): Whether to use distance-based codebook resets.
        """
        super(VQEmbedding, self).__init__()
        # Create an embedding layer (the codebook) with K embeddings of dimension D.
        self.vq_embs = nn.Embedding(codebook_size, n_dim)
        self.decay = decay
        self.unused_reset_threshold = unused_reset_threshold
        self.hot_reset_threshold = hot_reset_threshold
        self.distance_reset = distance_reset
        self.use_ema = use_ema
        # Initialize with normal distribution
        self.vq_embs.weight.data.normal_(0, 1.0)  # Using LayerNorm, we expect the norm to be 1.0
        
        # Register buffers for EMA updates
        self.register_buffer('cluster_size', torch.zeros(codebook_size), persistent=True)
        self.register_buffer('embed_sum', torch.zeros(codebook_size, n_dim), persistent=True)
        self.register_buffer('ema_initialized', torch.tensor(0, dtype=torch.bool), persistent=True)
        
        # Register buffer for tracking codebook changes
        self.register_buffer('previous_codebook', torch.zeros(codebook_size, n_dim), persistent=True)
        self.register_buffer('update_magnitudes', torch.zeros(codebook_size), persistent=True)
    
    def reset_unused_codes_random(self, z_e_x, normalized_embeddings, D, codebook_size):
        """
        Reset unused codebook entries based on low EMA counts.
        
        Args:
            z_e_x (Tensor): Encoder outputs, shape [B, N, D].
            normalized_embeddings (Tensor): The current normalized codebook embeddings, shape [K, D].
            D (int): Embedding dimension.
            codebook_size (int): Total number of codes (K).
            
        Returns:
            Tensor: Updated normalized_embeddings with unused entries replaced.
        """
        # Identify codebook entries with low usage.
        reset_mask = (self.cluster_size < self.unused_reset_threshold) | (self.cluster_size > self.hot_reset_threshold)  # Shape: [K]

        
        # Flatten encoder outputs to shape [B*N, D].
        flat_z_e = z_e_x.reshape(-1, D)
        
        # Sample random encoder outputs for all codebook entries.
        rand_idx = torch.randint(0, flat_z_e.shape[0], (codebook_size,), device=z_e_x.device)
        random_codes = flat_z_e[rand_idx]  # Shape: [K, D]
        
        # Replace normalized embeddings for unused entries using torch.where.
        normalized_embeddings = torch.where(
            reset_mask.unsqueeze(1),  # [K, 1]
            random_codes,
            normalized_embeddings
        )
        # Also update the EMA buffers for these entries.
        new_cluster_size = torch.where(
            reset_mask,
            torch.tensor(1.0, dtype=self.cluster_size.dtype, device=self.cluster_size.device),
            self.cluster_size
        )
        new_embed_sum = torch.where(
            reset_mask.unsqueeze(1),
            random_codes,
            self.embed_sum
        )
        self.cluster_size.copy_(new_cluster_size)
        self.embed_sum.copy_(new_embed_sum)
        return normalized_embeddings
    
    def reset_unused_codes_distance(self, z_e_x, normalized_embeddings, D, distances):
        """
        Reset unused codebook entries using quantization errors (distances) in a fully vectorized manner.
        For each codebook entry (of fixed size K), if its EMA count is below the threshold, it is updated
        with a candidate encoder output selected from those with the highest quantization error.
        
        Args:
            z_e_x (Tensor): Encoder outputs, shape [B, N, D].
            normalized_embeddings (Tensor): Normalized codebook embeddings, shape [K, D].
            D (int): Embedding dimension.
            distances (Tensor): Quantization errors, shape [B, N].
            
        Returns:
            Tensor: Updated normalized_embeddings.
        """

        with autocast(device_type='cuda', enabled=False):

            z_e_x_fp32 = z_e_x.float()
            normalized_embeddings_fp32 = normalized_embeddings.float()
            distances_fp32 = distances.float()
            # Fixed codebook size K.
            K = normalized_embeddings_fp32.shape[0]
            # Flatten encoder outputs and distances.
            flat_z_e = z_e_x_fp32.reshape(-1, D)      # [B*N, D]
            flat_dists = distances_fp32.reshape(-1)     # [B*N]
        
            # Sort encoder outputs by descending quantization error.
            sorted_indices = torch.argsort(flat_dists, descending=True)  # [B*N]
        
            # Compute a fixed-size unused mask (K is fixed).
            reset_mask = (self.cluster_size < self.unused_reset_threshold) | (self.cluster_size > self.hot_reset_threshold)
        
            # Compute a rank for each codebook entry: for unused entries, rank them in order of appearance.
            # Since K is fixed, this produces a fixed-shape tensor.
            unused_int = reset_mask.to(torch.int64)           # [K]
            ranks = torch.cumsum(unused_int, dim=0) - 1           # [K]
            # For used entries, force the rank to 0 (these values won't be used).
            ranks = torch.where(reset_mask, ranks, torch.zeros_like(ranks))
        
            # For each codebook entry i, candidate_codes[i] is taken from:
            # flat_z_e[ sorted_indices[ranks[i]] ]
            candidate_codes = flat_z_e[sorted_indices[ranks]]     # [K, D]
        
            # Update only unused entries in normalized_embeddings.
            updated_normalized_embeddings = torch.where(
                reset_mask.unsqueeze(1),  # shape [K, 1]
                candidate_codes,
                normalized_embeddings_fp32
            )
        
            # Update EMA buffers for the unused entries in a vectorized way.
            updated_cluster_size = torch.where(
                reset_mask,
                torch.tensor(1.0, dtype=self.cluster_size.dtype, device=self.cluster_size.device),
                self.cluster_size
            )
            updated_embed_sum = torch.where(
                reset_mask.unsqueeze(1),
                candidate_codes,
                self.embed_sum
            )
        
            self.cluster_size.copy_(updated_cluster_size)
            self.embed_sum.copy_(updated_embed_sum)
        
            return updated_normalized_embeddings.to(normalized_embeddings.dtype)
    
    def reset_unused_codes(self, z_e_x, normalized_embeddings, D, codebook_size, distances):
        if self.distance_reset:
            return self.reset_unused_codes_distance(z_e_x, normalized_embeddings, D, distances)
        else:
            return self.reset_unused_codes_random(z_e_x, normalized_embeddings, D, codebook_size)

    def update_codebook_ema(self, z_e_x, indices, distances):
        """
        Update codebook vectors using Exponential Moving Average (EMA).
        
        This method implements codebook updates using EMA as described in the VQ-VAE-2 paper.
        It then calls reset_unused_codes to reinitialize codes with low usage.
        
        Args:
            z_e_x (Tensor): Encoder output vectors [B, N, C]
                B = batch size
                N = number of codes per sample 
                C = embedding dimension
            indices (Tensor): Indices of nearest codebook entries [B, N]
            distances (Tensor): Quantization errors, shape [B, N].
        """
        with torch.no_grad(), autocast(device_type='cuda', enabled=False):
            # Store the current codebook for computing update magnitudes later
            self.previous_codebook.copy_(self.vq_embs.weight.data)
            
            # Get shapes
            B, N, D = z_e_x.shape  # [B, N, D]
            flat_idx = indices.reshape(B * N)  # [B*N]
            codebook_size = self.vq_embs.weight.shape[0]
            
            # Ensure same dtype for codebook and z_e_x.
            z_e_x_detached = z_e_x.detach().float()
            
            # Create one-hot encodings for indices.
            encodings = F.one_hot(flat_idx, num_classes=codebook_size).float()
            
            # Compute the new cluster size and sum of encoder outputs for each code.
            new_cluster_size = encodings.sum(0)  # [K]
            dw = encodings.t() @ z_e_x_detached.reshape(-1, D)  # [K, D]
            
            # ------------ TORCH.COMPILE FRIENDLY CODE ------------
            is_first_batch = ~self.ema_initialized  # scalar boolean tensor
            initialized_cluster_size = new_cluster_size  # [K]
            initialized_embed_sum = dw  # [K, D]
            
            updated_cluster_size = self.cluster_size.float() * self.decay + new_cluster_size * (1 - self.decay)
            updated_embed_sum = self.embed_sum.float() * self.decay + dw * (1 - self.decay)
            
            new_cluster_size = torch.where(is_first_batch, initialized_cluster_size, updated_cluster_size)
            new_embed_sum = torch.where(is_first_batch.unsqueeze(-1), initialized_embed_sum, updated_embed_sum)
            
            self.cluster_size.copy_(new_cluster_size.to(self.cluster_size.dtype))
            self.embed_sum.copy_(new_embed_sum.to(self.embed_sum.dtype))
            self.ema_initialized.copy_(torch.tensor(True, dtype=torch.bool, device=self.ema_initialized.device))
            # ------------ END TORCH.COMPILE FRIENDLY CODE ------------
            
            # Prevent division by zero.
            effective_cluster_size = self.cluster_size + 1e-3
            
            # Compute normalized embeddings.
            normalized_embeddings = (self.embed_sum / effective_cluster_size.unsqueeze(1))
            
            # Call the reset method to update unused codebook entries.
            normalized_embeddings = self.reset_unused_codes(z_e_x, normalized_embeddings, D, codebook_size, distances)
        
            # Update the codebook weights.
            self.vq_embs.weight.copy_(normalized_embeddings.to(self.vq_embs.weight.dtype))
        
            # Compute update magnitudes (L2 norm of the difference)
            updates = self.vq_embs.weight.data - self.previous_codebook
            self.update_magnitudes.copy_(torch.norm(updates, dim=1))


    def forward(self, z_e_x):
        """
        Forward pass with a straight-through estimator, optionally using EMA updates.
        
        This method quantizes the input vectors and allows gradients to flow through
        the quantization process by using the straight-through approach. It returns:
            - z_q_x: Quantized vectors with shape (B, N, C).
            - zqx_tilde: An alternative quantized representation derived directly by index selection,
                         also with shape (B, N, C).
            - idx: Quantization indices with shape (B, N).
        
        Args:
            z_e_x (Tensor): The continuous encoded output from the transformer.
                            Expected shape: (B, N, C), where C equals the embedding dimension D.
            use_ema (bool): Whether to use EMA updates for codebook.
            
        Returns:
            tuple: (z_q_x, zqx_tilde, idx)
        """
        # Input shape: (B, N, C)
        # Capture all outputs from the straight-through estimator.
        z_q_x, flat_idx, idx, distances = VQ_ST(z_e_x, self.vq_embs.weight.detach())
        
        # Retrieve alternative quantized representation using flat_idx.
        flat_zqx_tilde = torch.index_select(self.vq_embs.weight, dim=0, index=flat_idx)
        zqx_tilde = flat_zqx_tilde.view_as(z_e_x)

        # Always calculate both losses for monitoring
        vq_loss = F.mse_loss(zqx_tilde, z_e_x.detach())
        commitment_loss = F.mse_loss(z_e_x, zqx_tilde.detach())
        
        # Optionally update the codebook via EMA if enabled.
        if self.use_ema and self.training:
            self.update_codebook_ema(z_e_x, idx, distances)

        meta = {
            "vq_loss": vq_loss.detach() if self.use_ema else vq_loss,
            "commitment_loss": commitment_loss,
            "indices": idx
        }
        
        return z_q_x, meta


@dataclass
class CTDAutoEncoderConfig:
    n_vocab: int = 16
    n_emb: int = 16
    n_dim: int = 128
    n_layers: int = 3
    n_heads: int = 4
    codebook_size: int = 1024
    grid_height: int = 32
    grid_width: int = 32
    encode_norm: bool = True
    use_ema: bool = True
    decay: float = 0.99
    distance_reset: bool = True
    unused_reset_threshold: float = 1.0
    hot_reset_threshold: float = 500
    pad_weight: float = 1.0
    gamma: float = 2.0

    def __post_init__(self):
        # Ensure that out_channels / in_channels is a power of 2
        assert self.n_dim % self.n_emb == 0
        assert self.n_dim / self.n_emb == 2**int(math.log2(self.n_dim / self.n_emb))

        # Calculate the number of ConvBlocks needed
        num_blocks = int(math.log2(self.n_dim / self.n_emb))
        self.latent_height = self.grid_height // 2**num_blocks
        self.latent_width = self.grid_width // 2**num_blocks
        self.n_latent = self.latent_height * self.latent_width

        if self.codebook_size > 0:
            assert self.n_layers > 0, "Transformer encoder is required for Quantization "

        self.pad_idx = self.n_vocab - 1
        self.mask_idx = self.n_vocab - 2


class CTDAutoEncoder(nn.Module):
    def __init__(self, config: CTDAutoEncoderConfig):
        super(CTDAutoEncoder, self).__init__()
        self.embd = nn.Embedding(config.n_vocab, config.n_emb)
        self.latent_height = config.latent_height
        self.latent_width = config.latent_width
        self.n_latent = config.n_latent
        self.codebook_size = config.codebook_size
        self.pad_idx = config.pad_idx
        self.mask_idx = config.mask_idx
        self.pad_weight = config.pad_weight
        self.gamma = config.gamma
        self.conv_autoencoder = ConvAutoEncoder(config.n_emb, config.n_dim)
        self.trans_autoencoder = TransformerAutoEncoder(n_dim=config.n_dim, n_layers=config.n_layers, n_heads=config.n_heads, seq_len=config.n_latent, encode_norm=config.encode_norm)
        self.quantizer = VQEmbedding(codebook_size=config.codebook_size, n_dim=config.n_dim) if self.codebook_size > 0 else nn.Identity()
        self.out_proj = nn.Linear(config.n_emb, config.n_vocab, bias=False)

    def conv_encode(self, x):
        x = self.embd(x) # [batch, n_emb, grid_height, grid_width]
        x = x.permute(0, 3, 1, 2) # [batch, grid_height, grid_width, n_emb]
        x = self.conv_autoencoder.encode(x) # [batch, n_dim, latent_height, latent_width]
        return x
    
    def trans_encode(self, x):
        x = x.permute(0, 2, 3, 1) # [batch, latent_height, latent_width, n_dim]
        x = x.reshape(x.shape[0], -1, x.shape[-1]) # [batch, n_latent, n_dim]
        x = self.trans_autoencoder.encode(x) # [batch, n_latent, n_dim]
        return x
    
    def trans_decode(self, x):
        # x: [batch, n_latent, n_dim]
        x = self.trans_autoencoder.decode(x) # [batch, n_latent, n_dim]
        x = x.reshape(x.shape[0], self.latent_height, self.latent_width, x.shape[-1]) # [batch, latent_height, latent_width, n_dim]
        x = x.permute(0, 3, 1, 2) # [batch, n_dim, latent_height, latent_width]
        return x

    def conv_decode(self, x):
        x = self.conv_autoencoder.decode(x) # [batch, n_dim, grid_height, grid_width]
        x = x.permute(0, 2, 3, 1) # [batch, grid_height, grid_width, n_dim]
        x = self.out_proj(x) # [batch, grid_height, grid_width, n_vocab]
        return x

    def forward(self, x):
        x_conv = self.conv_encode(x) # [batch, n_dim, latent_height, latent_width]
        x_trans = self.trans_encode(x_conv) # [batch, n_latent, n_dim]

        meta_dict = {}
        if self.codebook_size > 0:
            x_trans, meta_dict = self.quantizer(x_trans) # [batch, n_latent, n_dim]

        x_trans = self.trans_decode(x_trans) # [batch, n_latent, n_dim]
        logits = self.conv_decode(x_trans) # [batch, grid_height, grid_width, n_vocab]
        meta_dict['ce_loss'] = self.reconstruction_loss(logits, x, pad_value=self.pad_idx, pad_weight=self.pad_weight, gamma=self.gamma)
        return x, meta_dict
    
    def reconstruction_loss(self, decoded_logits: torch.Tensor, x: torch.Tensor, pad_value: int = -1, pad_weight: float = 1.0, gamma: float = 0.0) -> torch.Tensor:
        """
        Compute the reconstruction loss using focal loss weighted cross-entropy per sample, with 
        padded tokens handled separately.

        When gamma == 0.0, the focal modulation factor is effectively 1 (i.e. standard cross-entropy loss).
        This avoids unnecessary computation.

        Args:
            decoded_logits (Tensor): Predicted logits of shape [B, S, V].
            x (Tensor): Target tokens of shape [B, S].
            pad_value (int): Token value for padding tokens.
            pad_weight (float): Weight for pad token loss. Valid tokens have weight 1.0.
            gamma (float): Focusing parameter for focal loss. If 0.0, no focal modulation is applied (default: 0.0).

        Returns:
            Tensor: Average reconstruction loss per sample.
        """
        # Create a weight tensor: assign pad tokens pad_weight, and valid tokens a weight of 1.0.
        weights = torch.where(
            x == pad_value,
            torch.full_like(x, pad_weight, dtype=decoded_logits.dtype),
            torch.ones_like(x, dtype=decoded_logits.dtype)
        )

        print("Weights shape: ", weights.shape)
        
        # Compute the standard cross-entropy loss per token without reduction.
        ce_loss = F.cross_entropy(
            decoded_logits.view(-1, decoded_logits.size(-1)),
            x.view(-1),
            reduction='none'
        )
        
        # If gamma > 0, apply focal loss modulation; otherwise, use the cross-entropy loss directly.
        if gamma > 0.0:
            # Estimate the probability for the true class: p_t ≈ exp(-ce_loss)
            p_t = torch.exp(-ce_loss)
            # Compute the focal modulation factor: (1 - p_t)^gamma.
            modulating_factor = (1 - p_t) ** gamma
            loss = modulating_factor * ce_loss
        else:
            loss = ce_loss
        
        # Compute the total loss: each token's loss is scaled by its corresponding weight.
        total_loss = (loss * weights.view(-1)).sum()
        # Normalize by the sum of static weights (this includes valid and pad tokens).
        total_weight = weights.sum()
        
        normalized_loss = total_loss / total_weight
        return normalized_loss


if __name__ == "__main__":
    config = CTDAutoEncoderConfig(n_layers=3, encode_norm=True, codebook_size=0)
    autoencoder = CTDAutoEncoder(config)
    print(autoencoder)

    n_emb = config.n_emb
    n_dim = config.n_dim
    n_vocab = config.n_vocab

    # Create a random input tensor of shape (1, n_emb, 32, 32) with values in [0, n_vocab)
    x = torch.randint(0, n_vocab, (1, 32, 32))
    print(x.shape)

    # Forward pass
    y, meta_dict = autoencoder(x)
    print(y.shape)
    print(meta_dict)    
