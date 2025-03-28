from dataclasses import dataclass
import math
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch import Tensor, nn
from typing import Optional, Tuple

from pips.dvae import Config, LatentTransformer, RoPE2D, RotaryPositionalEmbeddings, Transformer, is_power_of_two


import torch
from torch.autograd import Function


# Reference: https://github.com/Vrushank264/VQVAE-PyTorch/tree/main

class VectorQuantization(Function):
    """
    Custom autograd Function for vector quantization in a VQ-VAE.
    
    This function maps each input vector (of shape [B, N, C]) to the index of its closest
    embedding in the codebook (of shape [K, C]) using a squared L2 distance.
    
    Note: This function is non-differentiable; its backward pass is disabled.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens (or discrete codes)
      C = Embedding dimension
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Hard dimensions: inputs is expected to be [B, N, C]
        B, N, C = inputs.shape
        
        # Flatten the inputs to shape [B*N, C]
        flat_input = inputs.reshape(B * N, C)
        
        # Compute squared L2 norm of each codebook vector.
        # codebook: shape [K, C] --> codebook_sq: shape [K]
        codebook_sq = torch.sum(codebook * codebook, dim=1)
        
        # Compute squared L2 norm of each input vector.
        # flat_input: [B*N, C] --> inputs_sq: [B*N, 1]
        inputs_sq = torch.sum(flat_input * flat_input, dim=1, keepdim=True)
        
        # Compute squared Euclidean distance between each input and each codebook vector.
        # Using the identity: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*(x · e)
        # Here, torch.addmm computes: (codebook_sq + inputs_sq) + 2 * (flat_input @ codebook.T)
        # Since we're only selecting the minimum, the sign is unimportant.
        # Resulting shape: [B*N, K]
        l2_dis = torch.addmm(input=codebook_sq + inputs_sq,
                             mat1=flat_input,
                             mat2=codebook.t(),
                             alpha=2.0, beta=1.0)
        
        # For each input vector, find the index of the codebook vector with minimum distance.
        # idx_flat: shape [B*N]
        _, idx_flat = torch.min(l2_dis, dim=1)
        
        # Reshape indices back to hard dimensions [B, N]
        idx = idx_flat.reshape(B, N)
        
        # Mark these indices as non-differentiable.
        ctx.mark_non_differentiable(idx)
        
        return idx

    @staticmethod
    def backward(ctx, grad_outputs):
        raise RuntimeError("Backward pass is not defined for VectorQuantization. "
                           "Use VQStraightThrough instead.")

class VQStraightThrough(Function):
    """
    Custom autograd Function implementing the straight-through estimator for vector quantization.
    
    In the forward pass, it uses the above VectorQuantization (VQ) to get the nearest codebook
    indices, then retrieves the corresponding codebook embeddings (quantized vectors). In the
    backward pass, gradients are passed directly (straight-through) to the encoder, while gradients
    for the codebook are accumulated based on the quantization indices.
    
    Expected input shape: [B, N, C]
      B = Batch size
      N = Number of tokens
      C = Embedding dimension
    Codebook shape: [K, C]
    """
    @staticmethod
    def forward(ctx, inputs, codebook):
        # Hard dimensions: inputs is expected to be [B, N, C]
        B, N, C = inputs.shape
        
        # Get nearest codebook indices using VectorQuantization.
        idx = VectorQuantization.apply(inputs, codebook)  # idx has shape [B, N]
        
        # Flatten indices to shape [B*N] for later use.
        flat_idx = idx.reshape(B * N)
        
        # Save the flattened indices and the codebook for use in the backward pass.
        ctx.save_for_backward(flat_idx, codebook)
        ctx.mark_non_differentiable(flat_idx)
        
        # Retrieve quantized embeddings via index selection.
        # This gives a tensor of shape [B*N, C].
        codes_flat = torch.index_select(codebook, dim=0, index=flat_idx)
        
        # Reshape back to hard dimensions [B, N, C].
        codes = codes_flat.reshape(B, N, C)
        
        return codes, flat_idx

    @staticmethod
    def backward(ctx, grad_outputs, grad_indices):
        grad_inputs, grad_codebook = None, None
        
        # Pass the gradients to the encoder inputs directly using the straight-through method.
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.clone()
        
        
        # Compute gradients with respect to the codebook.
        if ctx.needs_input_grad[1]:
            
            # grad_outputs: gradient with respect to the quantized codes, shape [B, N, C]
            flat_idx, codebook = ctx.saved_tensors
        
            # Get the embedding dimension from the codebook.
            C = codebook.shape[1]

            # Flatten grad_outputs to shape [B*N, C] to match flat_idx.
            flat_grad_output = grad_outputs.reshape(-1, C)
            
            # Initialize gradient for the codebook with zeros.
            grad_codebook = torch.zeros_like(codebook)
            
            # Accumulate gradients for each codebook vector using the saved indices.
            grad_codebook.index_add_(0, flat_idx, flat_grad_output)
        
        return grad_inputs, grad_codebook


VQ = VectorQuantization.apply
VQ_ST =  VQStraightThrough.apply


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
    def __init__(self, K: int, D: int):
        """
        Initialize the VQEmbedding module.
        
        Args:
            K (int): Total number of embeddings in the codebook.
            D (int): Dimensionality of each embedding.
        """
        super(VQEmbedding, self).__init__()
        # Create an embedding layer (the codebook) with K embeddings of dimension D.
        self.vq_embs = nn.Embedding(K, D)
        # Initialize the codebook weights uniformly in the range [-1/K, 1/K]
        self.vq_embs.weight.data.uniform_(-1.0 / K, 1.0 / K)
        
    def forward(self, z_e_x):
        """
        Forward pass that maps continuous encoder outputs to discrete latent indices.
        
        Args:
            z_e_x (Tensor): The continuous encoded output from the transformer.
                            Expected shape: (B, N, C), where C equals the embedding dimension D.
        
        Returns:
            Tensor: Discrete latent indices with shape (B, N), where each element is an index
                    pointing to the closest embedding in the codebook.
        """
        # For transformer output, the shape is already (B, N, C) so no permutation is needed.
        # Use the vector quantization function (VQ) to get the nearest codebook indices.
        # VQ is the custom autograd function (VectorQuantization.apply).
        latents = VQ(z_e_x, self.vq_embs.weight)  # Resulting shape: (B, N)
        return latents
    
    def straight_through_forward(self, z_e_x):
        """
        Forward pass with a straight-through estimator.
        
        This method quantizes the input vectors and allows gradients to flow through
        the quantization process by using the straight-through approach. It returns:
            - z_q_x: Quantized vectors with shape (B, N, C).
            - zqx_tilde: An alternative quantized representation derived directly by index selection,
                         also with shape (B, N, C).
        
        Args:
            z_e_x (Tensor): The continuous encoded output from the transformer.
                            Expected shape: (B, N, C), where C equals the embedding dimension D.
        
        Returns:
            tuple: (z_q_x, zqx_tilde)
        """
        # Input shape: (B, N, C)
        # Apply the straight-through vector quantization.
        # VQ_ST is the straight-through variant (VQStraightThrough.apply) that allows gradient flow.
        # It returns quantized codes (z_q_x) and indices (idx) with shapes (B, N, C) and (B, N), respectively.
        z_q_x, idx = VQ_ST(z_e_x, self.vq_embs.weight.detach())
        
        # Flatten the indices for alternative representation computation.
        flat_idx = idx.view(-1)  # New shape: (B*N,)
        
        # Directly index the codebook using flat_idx to obtain the alternative quantized representation.
        # This returns a tensor of shape (B*N, D)
        flat_zqx_tilde = torch.index_select(self.vq_embs.weight, dim=0, index=flat_idx)
        
        # Reshape the flat tensor back to the input shape: (B, N, C)
        zqx_tilde = flat_zqx_tilde.view_as(z_e_x)  # New shape: (B, N, C)
        

        # The first output is the quantized codes and the second is the alternative quantized representation.
        # The first is straight-through and passed to the decoder.
        # The second has un-detached codebook weights and is used for the codebook/commitment loss.
        return z_q_x, zqx_tilde




@dataclass
class VQVAEConfig(Config):
    r"""
    Configuration class for VQVAE model.
    
    Attributes:
        n_dim (int): Model dimension
        n_head (int): Number of attention heads
        n_grid_layer (int): Number of base transformer layers
        n_latent_layer (int): Number of latent transformer layers
        n_codes (int): Number of discrete codes (must be power of 2)
        pos_dependent_codebook (bool): Whether to use position-dependent codebook (default: True)
        codebook_size (int): Size of codebook (default: 512)
        rope_base_height (int): Base for geometric progression in angle computation for height (default: 10007)
        rope_base_width (int): Base for geometric progression in angle computation for width (default: 5003)
        dropout (float): Dropout probability (default: 0.0)
        max_grid_height (int): Maximum grid height (default: 32)
        max_grid_width (int): Maximum grid width (default: 32)
        n_vocab (int): Vocabulary size (default: 16)
        padding_idx (int | None): Index for padding token (default: n_vocab - 1)
        mask_idx (int | None): Index for masking token (default: n_vocab - 2)
        pad_weight (float): Weight for pad token loss (default: 0.01)
        gamma (float): Focal loss gamma parameter. With \(\gamma=0\) there is no focal modulation, defaulting to 2.0.
    """
    n_dim: int = 256
    n_head: int = 4
    n_grid_layer: int = 2
    n_latent_layer: int = 2
    n_codes: int = 128
    codebook_size: int = 512
    rope_base_height: int = 10007  # ~10k, prime
    rope_base_width: int = 5003    # ~5k, prime
    dropout: float = 0.0
    max_grid_height: int = 32
    max_grid_width: int = 32
    n_vocab: int = 16
    padding_idx: int | None = None
    mask_idx: int | None = None
    pad_weight: float = 0.1
    gamma: float = 2.0
    init_mode: str = "normal"
    skip_codebook: bool = False

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        C = self.n_dim // self.n_head
        assert C % 2 == 0, "n_dim // n_head must be divisible by 2"

        head_dim = C // 2  # Actual Head Dimension
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Calculate n_pos from grid dimensions
        self.n_pos = self.max_grid_height * self.max_grid_width

        assert self.init_mode in ["normal", "xavier"], "Invalid initialization mode"
        
        assert is_power_of_two(self.n_pos), "Product of max_grid_height and max_grid_width must be a power of 2"
        # assert is_power_of_two(self.n_codes), "Number of codes must be a power of 2"
        # assert self.n_pos % self.n_codes == 0, "Number of positions must be divisible by the number of codes"

        # Set default padding_idx and eos_idx if not provided
        if self.padding_idx is None:
            self.padding_idx = self.n_vocab - 1
        if self.mask_idx is None:
            self.mask_idx = self.n_vocab - 2

    def __repr__(self) -> str:
        attrs = [f"{key}={getattr(self, key)}" for key in self.__annotations__.keys()]
        computed_attrs = [
            f"n_pos={self.n_pos}",
        ]
        all_attrs = attrs + computed_attrs
        return f"DVAEConfig({', '.join(all_attrs)})"

    def to_dict(self) -> dict:
        """Convert config to a dictionary.
        
        Returns:
            dict: Dictionary containing all config values, including computed attributes
        """
        base_dict = {
            'n_dim': self.n_dim,
            'n_head': self.n_head,
            'n_grid_layer': self.n_grid_layer,
            'n_latent_layer': self.n_latent_layer,
            'n_codes': self.n_codes,
            'codebook_size': self.codebook_size,
            'rope_base_height': self.rope_base_height,
            'rope_base_width': self.rope_base_width,
            'dropout': self.dropout,
            'max_grid_height': self.max_grid_height,
            'max_grid_width': self.max_grid_width,
            'n_vocab': self.n_vocab,
            'padding_idx': self.padding_idx,
            'mask_idx': self.mask_idx,
            'pad_weight': self.pad_weight,
            'gamma': self.gamma,
            'init_mode': self.init_mode,
            'skip_codebook': self.skip_codebook,
        }
        
        # Add computed attributes if they exist
        computed_attrs = ['n_pos']
        for attr in computed_attrs:
            if hasattr(self, attr):
                base_dict[attr] = getattr(self, attr)
                
        return base_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'VQVAEConfig':
        """Create config from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            VQVAEConfig: New config instance
            
        Raises:
            ValueError: If required fields are missing
        """
        # Extract only the fields that are part of the dataclass
        required_fields = cls.__annotations__.keys()
        config_kwargs = {
            key: config_dict[key] 
            for key in required_fields 
            if key in config_dict
        }
        
        # Create new instance
        return cls(**config_kwargs)

    def compute_latent_bits(self) -> float:
        """Calculate the theoretical information capacity of the discrete latent space in bits.
        
        The model encodes information using n_codes discrete variables, each with codebook_size 
        possible values. The total number of possible configurations is codebook_size^n_codes,
        which corresponds to log2(codebook_size^n_codes) = n_codes * log2(codebook_size) bits.
        
        Returns:
            float: The number of bits that can be encoded in the latent space
        """
        return self.n_codes * math.log2(self.codebook_size)


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.skip_codebook = config.skip_codebook
        self.gamma = config.gamma
        self.n_pos = config.n_pos
        self.embd = nn.Embedding(config.n_vocab, config.n_dim)
        self.pad_value = config.padding_idx  # Store padding value here
        self.mask_value = config.mask_idx
        nn.init.normal_(self.embd.weight, mean=0.0, std=0.02)
        

        ## The choice of out_norm is inspired by Llama. We can think of both Encoder and Decoder as Llama models.
        ## With the difference that some of the layers are replaced by TransformerProjection blocks.
        ## Like in Llama, the token embeddings flow through unnormalized until the head is applied.
        ## In my case, we normalise the final output of the encoder as well as that of decoder before applyin their
        ## respective heads. Nothing gets normalised from base to bottleneck and vice versa.

        rope_2d = RoPE2D(
                dim=config.n_dim // config.n_head,  # per-head dimension (e.g., 256//8 = 32)
                max_height=config.max_grid_height,
                max_width=config.max_grid_width,
                base_height=config.rope_base_height,
                base_width=config.rope_base_width)
        
        rope_1d = RotaryPositionalEmbeddings(
            dim=config.n_dim // config.n_head,  # 256//8 = 32, per-head dimension
            max_seq_len=config.n_pos,
            base=config.rope_base_height
        )

        
        self.grid_encoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=False,
            rope=rope_2d
        )

        self.latent_encoder = LatentTransformer(
            n_latent=config.n_codes,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,
            rope=rope_1d
        )


        self.codebook = VQEmbedding(config.n_codes, config.n_dim)    

        self.latent_decoder = LatentTransformer(
            n_latent=config.n_pos,
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_latent_layer,
            out_norm=False,
            rope=rope_1d
        )

        self.grid_decoder = Transformer(
            d_model=config.n_dim,
            n_head=config.n_head,
            n_layer=config.n_grid_layer,
            out_norm=True,
            rope=rope_2d
        )

        self.decoder_head = nn.Linear(config.n_dim, config.n_vocab, bias=False)

        rows = torch.arange(config.max_grid_height, dtype=torch.long)
        cols = torch.arange(config.max_grid_width, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        grid_pos_indices = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).unsqueeze(0)
        latent_pos_indices = torch.arange(config.n_pos).unsqueeze(0)

        self.register_buffer("latent_pos_indices", latent_pos_indices, persistent=False)
        self.register_buffer('grid_pos_indices', grid_pos_indices, persistent=False)

        # # Apply weight initialization on registered modules.
        # self.apply(self._init_weights)
        # # Additionally, initialize any raw nn.Parameters.
        # self.initialize_all_parameters()

    def _init_weights(self, module):
        # Get initialization mode from config (if present). Default is "normal".
        # Set self.config.init_mode = 'xavier' if you prefer Xavier initialization.
        init_mode = getattr(self.config, "init_mode", "normal")
        
        if isinstance(module, nn.Linear):
            if init_mode == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.weight._initialized = True
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                module.bias._initialized = True
        elif isinstance(module, nn.Embedding):
            if init_mode == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.weight._initialized = True
        elif isinstance(module, nn.LayerNorm):
            # Often LayerNorm weights are initialized to ones and biases to zeros.
            torch.nn.init.ones_(module.weight)
            module.weight._initialized = True
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                module.bias._initialized = True

    def initialize_all_parameters(self):
        """
        Initialize all parameters in the model recursively.
        This method is meant to also initialize raw nn.Parameter attributes that are not part
        of a submodule (and hence not handled by self.apply).
        """
        init_mode = getattr(self.config, "init_mode", "normal")
        for name, param in self.named_parameters():
            # If the parameter has already been initialized (flagged via _initialized), skip.
            if hasattr(param, "_initialized"):
                continue
            if param.ndim >= 2:
                if init_mode == "xavier":
                    torch.nn.init.xavier_normal_(param)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            param._initialized = True


    def encode(self, x: Tensor, grid_pos_indices: Tensor, latent_pos_indices: Tensor) -> Tensor:
        x_embd = self.embd(x) # [B, S, n_dim]
        grid_encoded, _ = self.grid_encoder(x_embd, positions=grid_pos_indices) # [B, S, n_dim]
        latent_encoded, _ = self.latent_encoder(grid_encoded, positions=latent_pos_indices) # [B, n_codes, n_dim]
        return latent_encoded


    def decode(self, x: Tensor, grid_pos_indices: Tensor, latent_pos_indices: Tensor) -> Tensor:    
        latent_decoded, _ = self.latent_decoder(x, positions=latent_pos_indices)        
        grid_decoded, _ = self.grid_decoder(latent_decoded, positions=grid_pos_indices)        
        grid_decoded_logits = self.decoder_head(grid_decoded)
        return grid_decoded_logits
    

    def apply_mask(self, x: Tensor, mask_percentage: Tensor) -> Tensor:
        x = x.clone()
        # Create a random tensor with values in the range [0,1) for each element in x.
        mask = (torch.rand(x.shape, device=x.device, dtype=torch.float32) < mask_percentage) & (x != self.pad_value)
        x.masked_fill_(mask, self.mask_value)
        return x
    

    def forward(self, x: Tensor, mask_percentage: Tensor = torch.tensor(0.0)) -> Tuple[Tensor, dict, Tensor]:
        B, S = x.size()
        x_masked = self.apply_mask(x, mask_percentage)
        grid_pos_indices = self.grid_pos_indices.expand(B, -1, -1)
        latent_pos_indices = self.latent_pos_indices.expand(B, -1)
    
        z_e_x = self.encode(x_masked, grid_pos_indices, latent_pos_indices) # [B, n_codes, n_dim]
                
        z_q_x_st, z_q_x = self.codebook.straight_through_forward(z_e_x) # [B, n_codes, n_dim]

        if self.skip_codebook:
            decoded_logits = self.decode(z_e_x, grid_pos_indices, latent_pos_indices)
        else:
            decoded_logits = self.decode(z_q_x_st, grid_pos_indices, latent_pos_indices)
    
        ce_loss = self.reconstruction_loss(decoded_logits, x, pad_value=self.pad_value, gamma=self.gamma)
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
        commitment_loss = F.mse_loss(z_e_x, z_q_x.detach())
    
        losses = {
             "ce_loss": ce_loss,  # Weight normalized loss
             "vq_loss": vq_loss,
             "commitment_loss": commitment_loss
        }
        return decoded_logits, losses


    def reconstruction_loss(self, decoded_logits: Tensor, x: Tensor, pad_value: int = -1, pad_weight: float = 1.0, gamma: float = 0.0) -> Tensor:
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
    