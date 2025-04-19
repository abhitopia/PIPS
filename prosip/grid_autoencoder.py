from dataclasses import asdict, dataclass
import math
import torch
import torch.nn as nn
from prosip.utils import get_activation_fn

def is_power_of_two(n):
    """Check if a number is a power of two.
    Works for both integers and floats."""
    # If n is very close to an integer, convert it
    if isinstance(n, float):
        # For floating point numbers, use a mathematical check
        # A power of 2 has log2(n) as an integer
        if n <= 0:
            return False
        return math.isclose(2**round(math.log2(n)), n, rel_tol=1e-9)
    else:
        # For integers, use the bitwise method which is more efficient
        return n > 0 and (n & (n - 1)) == 0





class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, stride=2, activation='gelu'):
        super(ResidualConvBlock, self).__init__()
        layers = []

        activation_module = get_activation_fn(activation, return_module=True)
        # First conv uses the provided stride (which may be 1 or 2)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation_module())
        # Additional convs use stride=1
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_module())
        self.block = nn.Sequential(*layers)
        # Adjust shortcut if spatial or channel dimensions change
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        return out + residual


class ResidualConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, stride=2, activation='gelu'):
        super(ResidualConvTransposeBlock, self).__init__()
        layers = []

        activation_module = get_activation_fn(activation, return_module=True)
        # First deconv uses the provided stride for upsampling.
        layers.append(nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, output_padding=1 if stride == 2 else 0))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation_module())
        # Additional convs with stride=1
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_module())
        self.block = nn.Sequential(*layers)
        # Shortcut: upsample using a 1x1 transposed conv.
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                padding=0, output_padding=1 if stride == 2 else 0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        return out + residual


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels, channels, input_resolution=32, final_resolution=4, num_blocks=4, num_convs=2, encode_norm=False, decode_norm=False, activation="gelu"):
        super(ConvAutoEncoder, self).__init__()
        # If num_blocks is 0, this will be an identity function
        if num_blocks == 0:
            self.initial_conv = nn.Identity()
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            self.encode_norm = nn.Identity()
            self.decode_norm = nn.Identity()
            self.adaptive_pool = nn.Identity()
            self.adaptive_upsample = None
            return
            
        self.input_resolution = input_resolution
        self.final_resolution = final_resolution

        # Use power-of-2 downsampling as an approximation, then adapt to exact size
        if is_power_of_two(input_resolution / final_resolution):
            # Original power-of-2 case
            required_downsampling = int(math.log2(input_resolution / final_resolution))
            use_adaptive = False
        else:
            # For non-power-of-2, find the closest power of 2 that's smaller or equal
            # This gives us a reasonable number of downsampling layers
            ratio = input_resolution / final_resolution
            required_downsampling = math.floor(math.log2(ratio))
            use_adaptive = True

        if num_blocks < required_downsampling:
            raise ValueError("num_blocks must be at least log2(input_resolution/final_resolution)")

        # Build strides list: use stride=2 for exactly 'required_downsampling' blocks, and stride=1 for the others.
        strides = [1] * num_blocks
        for i in range(num_blocks - required_downsampling, num_blocks):
            strides[i] = 2

        # If the input channels differ from our constant internal channels, add an initial conv to fix that.
        self.initial_conv = nn.Conv2d(in_channels, channels, kernel_size=1) if in_channels != channels else nn.Identity()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Build encoder blocks.
        for i in range(num_blocks):
            self.encoder.add_module(
                f"enc_block_{i}",
                ResidualConvBlock(channels, channels, num_convs=num_convs, stride=strides[i], activation=activation)
            )

        # Calculate the intermediate resolution after encoder blocks
        self.intermediate_resolution = input_resolution // (2 ** required_downsampling)
            
        # Add adaptive pooling to get exact resolution if needed
        if use_adaptive:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((final_resolution, final_resolution))
        else:
            self.adaptive_pool = nn.Identity()
            
        # For the decoder, we'll need to upsample from final_resolution to intermediate_resolution
        # if we used adaptive pooling
        if use_adaptive and self.intermediate_resolution != final_resolution:
            self.adaptive_upsample = nn.Upsample(
                size=(self.intermediate_resolution, self.intermediate_resolution),
                mode='bilinear',
                align_corners=False
            )
        else:
            self.adaptive_upsample = None

        # Build decoder blocks by reversing the strides order.
        decoder_strides = strides[::-1]
        for i in range(num_blocks):
            self.decoder.add_module(
                f"dec_block_{i}",
                ResidualConvTransposeBlock(channels, channels, num_convs=num_convs, stride=decoder_strides[i], activation=activation)
            )
            
        # Add RMSNorm for the encoded and decoded outputs if requested
        self.encode_norm = nn.RMSNorm(channels) if encode_norm else nn.Identity()
        self.decode_norm = nn.RMSNorm(channels) if decode_norm else nn.Identity()

    def encode(self, x):
        x = self.initial_conv(x)
        x = self.encoder(x)
        
        # Apply adaptive pooling if needed
        x = self.adaptive_pool(x)
        
        # Apply RMSNorm if requested (need to reshape for RMSNorm then reshape back)
        if not isinstance(self.encode_norm, nn.Identity):
            # Reshape from [B, C, H, W] to [B, H, W, C] for RMSNorm
            x_norm = x.permute(0, 2, 3, 1)
            x_norm = self.encode_norm(x_norm)
            # Reshape back to [B, C, H, W]
            x = x_norm.permute(0, 3, 1, 2)
        
        return x

    def decode(self, x):
        # Apply adaptive upsampling if needed
        if self.adaptive_upsample is not None:
            x = self.adaptive_upsample(x)
            
        x = self.decoder(x)
        
        # Apply RMSNorm if requested (need to reshape for RMSNorm then reshape back)
        if not isinstance(self.decode_norm, nn.Identity):
            # Reshape from [B, C, H, W] to [B, H, W, C] for RMSNorm
            x_norm = x.permute(0, 2, 3, 1)
            x_norm = self.decode_norm(x_norm)
            # Reshape back to [B, C, H, W]
            x = x_norm.permute(0, 3, 1, 2)
            
        return x
    
    def forward(self, x):
        return self.decode(self.encode(x))


class TransformerAutoEncoder(nn.Module):
    def __init__(self, n_dim, n_layers, n_heads, seq_len, dropout=0.0, encode_norm=True, decode_norm=False, activation="relu"):
        super(TransformerAutoEncoder, self).__init__()
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len

        # Position embedding with scaled initialization
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.n_dim) * 0.02) if self.n_layers > 0 else torch.zeros(1, self.seq_len, self.n_dim, requires_grad=False)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.n_dim, 
                                       nhead=self.n_heads, 
                                       dim_feedforward=4*self.n_dim, 
                                       dropout=dropout,
                                       batch_first=True, bias=False, norm_first=True, 
                                       activation=get_activation_fn(activation, return_module=False)),
            num_layers=self.n_layers,
            enable_nested_tensor=False
        ) if self.n_layers > 0 else nn.Identity()

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.n_dim, 
                                       nhead=self.n_heads, 
                                       dim_feedforward=4*self.n_dim, 
                                       dropout=dropout,
                                       batch_first=True, bias=False, norm_first=True, 
                                       activation=get_activation_fn(activation, return_module=False)),
            num_layers=self.n_layers,
            enable_nested_tensor=False
        ) if self.n_layers > 0 else nn.Identity()

        self.encode_norm = nn.RMSNorm(self.n_dim) if encode_norm else nn.Identity()
        self.decode_norm = nn.RMSNorm(self.n_dim) if decode_norm else nn.Identity()

    def encode(self, x):
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.encode_norm(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        x = self.decode_norm(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    

@dataclass
class GridAutoEncoderConfig:
    n_vocab: int = 16
    n_dim: int = 256
    n_layer: int = 4
    n_head: int = 4
    latent_height: int = 8
    latent_width: int = 8
    conv_block_size: int = 2
    n_conv_blocks: int = 2
    grid_height: int = 32
    grid_width: int = 32
    encode_norm: bool = True
    decode_norm: bool = True
    activation: str = "gelu"
    dropout: float = 0.0

    def __post_init__(self):
        # Ensure that out_channels / in_channels is a power of 2
        # assert self.n_dim % self.n_emb == 0
        # assert self.n_dim / self.n_emb == 2**int(math.log2(self.n_dim / self.n_emb))

        # assert self.grid_height % self.latent_height == 0, "grid_height must be divisible by latent_height"
        # assert self.grid_width % self.latent_width == 0, "grid_width must be divisible by latent_width"

        # assert is_power_of_two(self.latent_height), "latent_height must be a power of 2"
        # assert is_power_of_two(self.latent_width), "latent_width must be a power of 2"

        self.n_codes = self.latent_height * self.latent_width
        self.latent_resolution = self.latent_height

        self.pad_idx = self.n_vocab - 1
        self.mask_idx = self.n_vocab - 2

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


class GridAutoEncoder(nn.Module):
    def __init__(self, config: GridAutoEncoderConfig):
        super(GridAutoEncoder, self).__init__()
        self.config = config
        self.mask_idx = config.mask_idx
        self.pad_idx = config.pad_idx
        self.embd = nn.Embedding(config.n_vocab, config.n_dim)
        self.latent_height = config.latent_height
        self.latent_width = config.latent_width
        self.n_latent = config.n_codes
        self.conv_autoencoder = ConvAutoEncoder(in_channels=config.n_dim, channels=config.n_dim, 
                                                input_resolution=32, 
                                                final_resolution=self.latent_height,
                                                num_blocks=config.n_conv_blocks,
                                                num_convs=config.conv_block_size,
                                                encode_norm=config.encode_norm,
                                                decode_norm=config.decode_norm,
                                                activation=config.activation)
        
        self.trans_autoencoder = TransformerAutoEncoder(n_dim=config.n_dim, n_layers=config.n_layer, n_heads=config.n_head, 
                                                        seq_len=config.n_codes, 
                                                        dropout=config.dropout,
                                                        encode_norm=config.encode_norm,
                                                        decode_norm=config.decode_norm,
                                                        activation=config.activation)
        
        
        self.out_proj = nn.Linear(config.n_dim, config.n_vocab, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all parameters following standard practices for this architecture.
        - Embeddings: Normal distribution with std=0.02
        - Linear layers: Normal distribution with std=0.02
        - Conv/ConvTranspose layers: Normal distribution with std=0.02
        - BatchNorm: Default PyTorch initialization (weight=1, bias=0)
        - RMSNorm: Weight=1.0
        - Position embeddings in Transformer: Already initialized in TransformerAutoEncoder
        """
        
        def init_weights(module):
            # Skip if we've already processed this module
            if isinstance(module, nn.Linear):
                # Initialize linear weights with normal distribution
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Initialize conv weights with normal distribution
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm uses ones for weights and zeros for bias
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.RMSNorm):
                # Initialize layer norm parameters
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                # Initialize embedding weights
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Only call reset_parameters on PyTorch built-in modules, not our custom modules
            elif hasattr(module, 'reset_parameters') and module is not self:
                module.reset_parameters()
            else:
                # Handle custom parameters that are directly attached to the module
                for name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        if param.dim() > 1:
                            # For matrices/tensors, use normal initialization
                            nn.init.normal_(param, mean=0.0, std=0.02)
                        else:
                            # For vectors (1D tensors), initialize to zeros
                            # except for special cases like RMSNorm weights
                            if 'norm' in name and 'weight' in name:
                                nn.init.ones_(param)
                            else:
                                nn.init.zeros_(param)
        
        # First initialize our own parameters (not recursively)
        self.apply(init_weights)

    def conv_encode(self, x):
        x = x.permute(0, 3, 1, 2) # [batch, n_emb, grid_height, grid_width] 
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
    
    def apply_mask(self, x: torch.Tensor, mask_percentage: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        # Create a random tensor with values in the range [0,1) for each element in x.
        mask = (torch.rand(x.shape, device=x.device, dtype=torch.float32) < mask_percentage) & (x != self.pad_idx)
        x.masked_fill_(mask, self.mask_idx)
        return x
    
    def encode(self, x, mask_percentage: torch.Tensor = torch.tensor(0.0)):
        x_masked = self.apply_mask(x, mask_percentage)
        x_masked = self.embd(x_masked) # [batch, grid_height, grid_width, n_emb]
        x_conv = self.conv_encode(x_masked) # [batch, n_dim, latent_height, latent_width]
        x_trans = self.trans_encode(x_conv) # [batch, n_latent, n_dim]
        return x_trans
    
    def decode(self, x):
        x_trans = self.trans_decode(x) # [batch, n_latent, n_dim]
        logits = self.conv_decode(x_trans) # [batch, n_dim, grid_height, grid_width]
        return logits

    def forward(self, x, mask_percentage: torch.Tensor = torch.tensor(0.0)):
        x_trans = self.encode(x, mask_percentage) # [batch, n_latent, n_dim]
        logits = self.decode(x_trans) # [batch, n_dim, grid_height, grid_width]
        return logits
    
