import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
import argparse
from pathlib import Path
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from pips.dvae import RMSNorm, RotaryPositionalEmbeddings, SwiGLUFFN

    
class AttnCodebook(nn.Module):
    """
    A single-head attention-based codebook module.
    
    The latent encoder outputs are treated as queries, and the codebook consists
    of learned keys and values. This allows the module to compute an attention 
    weighted sum over the codebook entries.
    
    Args:
        d_model (int): The model dimension.
        codebook_size (int): The number of codebook entries.
    
    Shapes:
        - queries: [B, N, d_model] where B=batch size, N=number of tokens.
        - codebook_keys: [d_model, codebook_size]
        - codebook_values: [codebook_size, d_model]
    """
    def __init__(self, d_model: int, codebook_size: int, use_exp_relaxed=False, sampling: bool = True, dim_feedforward=None, rope=None, normalise_kq: bool = False):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.use_exp_relaxed = use_exp_relaxed
        self.sampling = sampling
        self.rope = rope
        self.normalise_kq = normalise_kq

        self.norm_context = RMSNorm(dim=d_model)
        self.norm_queries = RMSNorm(dim=d_model)

        # Shared codebook embeddings: shape [codebook_size, d_model]
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model))

        ## Attention Stuff
        # Projection layers to generate keys and values from the codebook.
        # These layers project the codebook from [codebook_size, d_model] to the same shape.
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.d_model))
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        ## Feedforward Stuff
        self.ff = SwiGLUFFN(dim=self.d_model, hidden_dim=dim_feedforward if dim_feedforward is not None else 4*self.d_model)
        self.norm_ff = RMSNorm(dim=d_model)


    def sample(self, log_alpha: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Sample from either RelaxedOneHotCategorical or ExpRelaxedCategorical."""
        assert tau > 0.0, "Temperature must be greater than 0.0"
        # tau is already a tensor, so we simply use it directly.
        if self.use_exp_relaxed:
            # We need to exponentiate the sample to get the correct sample from the distribution.
            return ExpRelaxedCategorical(tau, logits=log_alpha).rsample().exp()
        else:
            return RelaxedOneHotCategorical(tau, logits=log_alpha).rsample()
    

    def single_head_attention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, tau: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Single-head attention pass.
        """
        # Project queries: [B, N, d_model]
        q = self.query_proj(queries).unsqueeze(1) # [B, 1, N, d_model]
        k = self.key_proj(keys).unsqueeze(0).expand(q.size(0), -1, -1).unsqueeze(1) # [B, 1, C, d_model]
        v = self.value_proj(values).unsqueeze(0).expand(q.size(0), -1, -1) # [B, C, d_model]

        qT = q.size(2)
        kT = k.size(2)

        q = self.rope(q, positions[:, :qT].unsqueeze(1)).squeeze(1)
        k = self.rope(k, positions[:, :kT].unsqueeze(1)).squeeze(1)


        if self.normalise_kq:
            ## Normalise the keys and queries so network doesn't fight back the decrease in temperature
            q = q / torch.norm(q, dim=-1, keepdim=True)
            k = k / torch.norm(k, dim=-1, keepdim=True)

        # Compute scaled dot-product attention.
        log_alpha = torch.matmul(q, k.mT) # [B, N, C]
        log_alpha = log_alpha / self.scale # [B, N, C]
        
        if self.sampling:
            # Sample using the (Gumbel-Softmax) distribution.
            z = self.sample(log_alpha, tau)  # [B, N, C]
        else:
            # Apply softmax with temperature scaling.
            z = torch.softmax(log_alpha / tau, dim=-1)

        attn_output = torch.matmul(z, v) # [B, N, d_model]
        y = self.c_proj(attn_output) # [B, N, d_model]

        return y, log_alpha, z
    

    def forward(self, latents: torch.Tensor, tau: torch.Tensor, residual_scaling: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the codebook.

        Args:
            latents (Tensor): The input latents.
            tau (Tensor): The temperature for the Gumbel-Softmax distribution.
            residual_scaling (Tensor): The scaling factor for the residual connection.
            positions (Tensor): The positions of the latents.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple of (codebook_output, log_alpha, z), where:
                codebook_output (Tensor): The output of the codebook.
                log_alpha (Tensor): The unnormalized logits used for distribution over the codebook.
                z (Tensor): The soft assignment code (either sampled or computed deterministically via softmax).
        """

        normed_context = self.norm_context(self.codebook) # [C, d_model]
        normed_queries = self.norm_queries(latents) # [B, N, d_model]

        attn_output, log_alpha, z = self.single_head_attention(
                                                            queries=normed_queries, 
                                                            keys=normed_context, 
                                                            values=normed_context, 
                                                            tau=tau,
                                                            positions=positions)

        # Notice that this mixes the latents with the codebook embeddings.
        # But this also means that attn_output is not a function of log_alpha only.
        # As such, we add a residual scaling factor to the residual connection.
        attn_output = residual_scaling * latents + attn_output # Residual connection

        codebook_output = attn_output + self.ff(self.norm_ff(attn_output)) # Residual connection

        return codebook_output, log_alpha, z



class Encoder(nn.Module):
    """
    Encoder network for MNIST images.
    Maps 28x28 images to a latent representation with shape [B, N, D]
    where B is batch size, N is number of codes, and D is model dimension.
    """
    def __init__(self, latent_dim, n_codes, hidden_dim=400):
        super().__init__()
        self.n_codes = n_codes
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_codes * latent_dim)
        
    def forward(self, x):
        # Input: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [B, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [B, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Reshape to [B, N, D]
        x = x.view(-1, self.n_codes, self.latent_dim)
        return x
    



class Decoder(nn.Module):
    """
    Decoder network for MNIST images.
    Maps latent representation [B, N, D] back to 28x28 images.
    """
    def __init__(self, latent_dim, n_codes, hidden_dim=400):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_codes = n_codes
        
        self.fc1 = nn.Linear(n_codes * latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # Input: [B, N, D]
        # Flatten the N and D dimensions
        x = x.view(-1, self.n_codes * self.latent_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # Sigmoid to get values in [0, 1]
        return x

class DiscreteVAE(nn.Module):
    """
    Discrete VAE model for MNIST using AttnCodebook for discretization.
    """
    def __init__(self, latent_dim, codebook_size, n_codes=8, hidden_dim=400, 
                 use_exp_relaxed=False, sampling=True):
        super().__init__()
        self.encoder = Encoder(latent_dim, n_codes, hidden_dim)
        self.rope = RotaryPositionalEmbeddings(latent_dim)
        self.codebook = AttnCodebook(d_model=latent_dim, codebook_size=codebook_size, 
                                     use_exp_relaxed=use_exp_relaxed, sampling=sampling,
                                     rope=self.rope)
        self.decoder = Decoder(latent_dim, n_codes, hidden_dim)

        self.latent_pos_indices = torch.arange(1024).unsqueeze(0)

        
    def forward(self, x, tau):
        # Encode input to shape [B, N, D]
        latent = self.encoder(x)
        
        latent_pos_indices = self.latent_pos_indices.expand(x.size(0), -1)
        
        # Discretize through codebook
        quantized, log_alpha, z = self.codebook(
                        latents=latent, 
                        tau=tau, 
                        residual_scaling=0.0, 
                        positions=latent_pos_indices)
        
        # Decode
        recon = self.decoder(quantized)
        
        return recon, log_alpha, z, latent, quantized

def entropy_loss(log_alpha, tau, reduction="mean"):
    """
    Compute the entropy of the latent distribution, accounting for temperature.
    
    Args:
        log_alpha: Logits for the latent distribution, shape [B, N, codebook_size]
        tau: Temperature parameter for scaling logits
        reduction: Reduction method ('sum', 'mean', or 'batchmean')
                
    Returns:
        Entropy reduced according to the specified method
    """
    # Apply temperature to logits
    scaled_log_alpha = log_alpha / tau
    
    # Compute log probabilities using log_softmax
    log_probs = F.log_softmax(scaled_log_alpha, dim=-1)
    
    # Get probabilities by exponentiating log probabilities
    probs = torch.exp(log_probs)
    
    # Compute entropy: -sum(p * log(p))
    entropy_per_sample = -torch.sum(probs * log_probs, dim=-1)  # [B, N]
    
    # Apply reduction
    if reduction == "sum":
        return entropy_per_sample.sum()
    elif reduction == "mean":
        return entropy_per_sample.mean()
    elif reduction == "batchmean":
        return entropy_per_sample.mean()
    elif reduction == "none":
        return entropy_per_sample
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def codebook_diversity_loss(log_alpha, tau, reduction="mean"):
    """
    Compute a diversity loss that encourages different samples to use different codebook entries.
    
    Args:
        log_alpha: Logits for the latent distribution, shape [B, N, codebook_size]
        tau: Temperature parameter for scaling logits
        reduction: Reduction method ('sum', 'mean', or 'batchmean')
                
    Returns:
        Diversity loss reduced according to the specified method
    """
    # Apply temperature to logits
    scaled_log_alpha = log_alpha / tau
    
    # Compute probabilities
    probs = F.softmax(scaled_log_alpha, dim=-1)  # [B, N, codebook_size]
    
    # Average usage of each codebook entry across the batch
    # This gives us the average probability of each codebook entry for each code position
    batch_avg_probs = probs.mean(dim=0)  # [N, codebook_size]
    
    # Compute entropy of the batch-averaged distribution
    # High entropy means different samples use different codebook entries
    # Low entropy means all samples use the same codebook entries
    log_batch_avg_probs = torch.log2(batch_avg_probs + 1e-10)
    batch_entropy = -torch.sum(batch_avg_probs * log_batch_avg_probs, dim=-1)  # [N]
    
    # We want to maximize this entropy, so we negate it for minimization
    diversity_loss = -batch_entropy  # [N]
    
    # Apply reduction
    if reduction == "sum":
        return diversity_loss.sum()
    elif reduction == "mean":
        return diversity_loss.mean()
    elif reduction == "batchmean":
        return diversity_loss.mean()
    elif reduction == "none":
        return diversity_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def compute_perplexity(log_alpha, tau):
    """
    Compute perplexity of probability distributions, accounting for temperature.
    Perplexity = 2^(entropy), measures how uniform the distribution is.
    
    Args:
        log_alpha: Logits for the latent distribution [B, N, codebook_size]
        tau: Temperature parameter for scaling logits
        
    Returns:
        Perplexity per sample [B, N]
    """
    # Apply temperature to logits and compute probabilities
    probs = F.softmax(log_alpha / tau, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log2(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, N]
    
    # Perplexity = 2^(entropy)
    perplexity = 2.0 ** entropy
    
    return perplexity

def compute_peakiness(log_alpha, tau):
    """
    Compute peakiness (max probability) of distributions, accounting for temperature.
    
    Args:
        log_alpha: Logits for the latent distribution [B, N, codebook_size]
        tau: Temperature parameter for scaling logits
        
    Returns:
        Peakiness per sample [B, N]
    """
    # Apply temperature to logits and compute probabilities
    probs = F.softmax(log_alpha / tau, dim=-1)
    return probs.max(dim=-1)[0]  # [B, N]

def visualize_reconstructions(original, recon, n_samples=8, save_path=None):
    """
    Visualize original and reconstructed MNIST images.
    
    Args:
        original: Original images [B, 1, 28, 28]
        recon: Reconstructed images [B, 1, 28, 28]
        n_samples: Number of samples to visualize
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 4))
    
    for i in range(n_samples):
        # Original image
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(original[i, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        
        # Reconstructed image
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(recon[i, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_code_distributions(log_alpha, tau, avg_peakiness=None, avg_perplexity=None, save_path=None):
    """
    Visualize probability distributions for each code position, accounting for temperature.
    
    Args:
        log_alpha: Logits for the latent distribution [B, N, codebook_size]
        tau: Temperature parameter for scaling logits
        avg_peakiness: Average peakiness for each code position [N] (optional)
        avg_perplexity: Average perplexity for each code position [N] (optional)
        save_path: Path to save the figure (optional)
    """
    # Apply temperature to logits and compute probabilities
    probs = F.softmax(log_alpha / tau, dim=-1)
    
    n_samples = min(2, probs.size(0))  # Show at most 2 samples
    n_codes = probs.size(1)  # Show all codes
    
    # Calculate the number of rows and columns for the subplot grid
    n_rows = n_samples
    n_cols = n_codes
    
    fig = plt.figure(figsize=(max(15, n_cols * 3), n_rows * 3))
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    # Find the global max probability for consistent y-axis scaling
    max_prob = probs.max().item()
    
    for sample_idx in range(n_samples):
        for code_idx in range(n_codes):
            ax = plt.subplot(gs[sample_idx, code_idx])
            
            # Get probabilities for this sample and code position
            probs_to_plot = probs[sample_idx, code_idx].detach().cpu().numpy()
            
            # Calculate peakiness and perplexity for this sample
            peakiness = probs[sample_idx, code_idx].max().item()
            entropy = -torch.sum(probs[sample_idx, code_idx] * torch.log2(probs[sample_idx, code_idx] + 1e-10)).item()
            perplexity = 2.0 ** entropy
            
            # Get indices where probability is non-zero (or above a small threshold)
            # This ensures we at least plot the highest probability even if it's very dominant
            threshold = 1e-6
            indices = np.where(probs_to_plot > threshold)[0]
            
            # If no indices meet the threshold, at least show the max value
            if len(indices) == 0:
                max_idx = np.argmax(probs_to_plot)
                indices = np.array([max_idx])
            
            # Plot distribution - only for indices with non-negligible probability
            x_positions = np.arange(len(probs_to_plot))
            bars = ax.bar(x_positions, probs_to_plot)
            
            # Highlight the highest bar in a different color
            max_idx = np.argmax(probs_to_plot)
            bars[max_idx].set_color('red')
            
            # Set y-axis limit to global max with 5% padding
            ax.set_ylim(0, max_prob * 1.05)
            
            # Add title with average metrics if provided
            if sample_idx == 0:
                title = f"Code {code_idx+1}"
                if avg_peakiness is not None and avg_perplexity is not None:
                    title += f"\nAvg Peak: {avg_peakiness[code_idx]:.3f}, Perp: {avg_perplexity[code_idx]:.2f}"
                ax.set_title(title)
            
            if code_idx == 0:
                ax.set_ylabel(f"Sample {sample_idx+1}\nProb")
            
            # Add sample-specific peakiness and perplexity as text
            ax.text(0.5, 0.02, f"Peak: {peakiness:.3f}\nPerp: {perplexity:.2f}", 
                    transform=ax.transAxes, ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Add text annotation for the max value
            if peakiness > 0.5:  # Only annotate high peaks
                ax.annotate(f"{peakiness:.3f}", 
                           xy=(max_idx, peakiness),
                           xytext=(max_idx, peakiness + max_prob * 0.05),
                           ha='center',
                           arrowprops=dict(arrowstyle='->'))
            
            if sample_idx == n_samples - 1:
                ax.set_xlabel("Codebook Index")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_experiment(args):
    """
    Run an experiment to study discrete representation learning on MNIST.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create a unique experiment folder based on arguments and a random suffix
    import uuid
    import time
    
    # Generate a short random suffix
    random_suffix = str(uuid.uuid4())[:8]
    
    # Create a descriptive experiment name
    exp_name = f"dim{args.latent_dim}_codes{args.n_codes}_cb{args.codebook_size}"
    exp_name += f"_temp{args.min_temp}-{args.max_temp}_ent{args.min_entropy_weight}-{args.max_entropy_weight}"
    if args.diversity_weight > 0:
        exp_name += f"_div{args.diversity_weight}"
    if args.use_exp_relaxed:
        exp_name += "_exprelaxed"
    if args.sampling:
        exp_name += "_sampling"
    exp_name += f"_{random_suffix}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the arguments for reproducibility
    with open(output_dir / "args.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Create subdirectories
    plots_dir = output_dir / "plots"
    recon_dir = plots_dir / "reconstructions"
    dist_dir = plots_dir / "distributions"
    metrics_dir = plots_dir / "metrics"
    
    plots_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)
    dist_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Keep the directory to "./data" as it is already in the .gitignore
    train_dataset = datasets.MNIST('./.data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./.data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = DiscreteVAE(
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        n_codes=args.n_codes,
        hidden_dim=args.hidden_dim,
        use_exp_relaxed=args.use_exp_relaxed,
        sampling=args.sampling
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'recon_loss': [],
        'entropy_loss': [],
        'temperature': [],
        'entropy_weight': [],
        'avg_peakiness': [],
        'avg_perplexity': [],
        'code_peakiness': [],  # Will store peakiness for each code position
        'code_perplexity': [],  # Will store perplexity for each code position
        'steps': []  # Track the step numbers
    }
    
    # Temperature annealing schedule
    if args.temp_anneal_steps > 0:
        temp_schedule = lambda step: max(args.min_temp, 
                                        args.max_temp - (args.max_temp - args.min_temp) * 
                                        min(1.0, step / args.temp_anneal_steps))
    else:
        temp_schedule = lambda step: args.min_temp
    
    # Entropy weight annealing schedule
    if args.entropy_anneal_steps > 0:
        entropy_schedule = lambda step: min(args.max_entropy_weight,
                                           args.min_entropy_weight + (args.max_entropy_weight - args.min_entropy_weight) * 
                                           min(1.0, step / args.entropy_anneal_steps))
    else:
        entropy_schedule = lambda step: args.max_entropy_weight
    
    # Training loop
    global_step = 0
    epoch = 0
    
    # Track peakiness and perplexity
    epoch_peakiness = []
    epoch_perplexity = []
    
    # Initialize per-code metrics with the correct shape [n_codes]
    code_peakiness_sum = np.zeros(args.n_codes)
    code_perplexity_sum = np.zeros(args.n_codes)
    code_counts = 0
    
    # Progress bar for total steps
    progress_bar = tqdm(total=args.max_steps, desc="Training")
    
    while global_step < args.max_steps:
        model.train()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            if global_step >= args.max_steps:
                break
                
            # Set temperature and entropy weight for this step
            temperature = temp_schedule(global_step)
            entropy_weight = entropy_schedule(global_step)
            
            # Convert temperature to tensor
            temp_tensor = torch.tensor(temperature)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, log_alpha, z, _, _ = model(data, temp_tensor)
            
            # Compute losses - using mean reconstruction loss
            recon_loss = F.binary_cross_entropy(recon, data, reduction='mean')
            ent_loss = entropy_loss(log_alpha, temp_tensor, reduction="mean")
            
            # Compute diversity loss if weight > 0
            if args.diversity_weight > 0:
                div_loss = codebook_diversity_loss(log_alpha, temp_tensor, reduction="mean")
                # Total loss - ADDING entropy loss to encourage discrete selection and diversity loss
                loss = recon_loss + entropy_weight * ent_loss + args.diversity_weight * div_loss
            else:
                div_loss = torch.tensor(0.0)
                # Total loss - ADDING entropy loss to encourage discrete selection
                loss = recon_loss + entropy_weight * ent_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate peakiness and perplexity for each code position
            batch_peakiness = compute_peakiness(log_alpha, temp_tensor).detach().cpu().numpy()  # [B, N]
            batch_perplexity = compute_perplexity(log_alpha, temp_tensor).detach().cpu().numpy()  # [B, N]
            
            # Average over batch dimension for each code position
            avg_batch_peakiness = np.mean(batch_peakiness, axis=0)  # [N]
            avg_batch_perplexity = np.mean(batch_perplexity, axis=0)  # [N]
            
            # Store the average peakiness and perplexity across all codes
            epoch_peakiness.append(np.mean(avg_batch_peakiness))
            epoch_perplexity.append(np.mean(avg_batch_perplexity))
            
            # Accumulate per-code metrics
            code_peakiness_sum += avg_batch_peakiness  # [N]
            code_perplexity_sum += avg_batch_perplexity  # [N]
            code_counts += 1
            
            # Update progress bar
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'entropy': ent_loss.item(),
                'diversity': div_loss.item() if args.diversity_weight > 0 else 0.0,
                'temp': temperature,
                'ent_w': entropy_weight
            })
            
            # Validation check at specified intervals
            if global_step % args.val_check_interval == 0 or global_step == args.max_steps:
                # Calculate average metrics
                avg_loss = loss.item()
                avg_recon_loss = recon_loss.item()
                avg_entropy_loss = ent_loss.item()
                avg_peakiness = np.mean(epoch_peakiness)
                avg_perplexity = np.mean(epoch_perplexity)
                
                # Calculate per-code metrics
                avg_code_peakiness = code_peakiness_sum / max(1, code_counts)
                avg_code_perplexity = code_perplexity_sum / max(1, code_counts)
                
                # Store metrics
                metrics['steps'].append(global_step)
                metrics['train_loss'].append(avg_loss)
                metrics['recon_loss'].append(avg_recon_loss)
                metrics['entropy_loss'].append(avg_entropy_loss)
                metrics['temperature'].append(temperature)
                metrics['entropy_weight'].append(entropy_weight)
                metrics['avg_peakiness'].append(avg_peakiness)
                metrics['avg_perplexity'].append(avg_perplexity)
                metrics['code_peakiness'].append(avg_code_peakiness.tolist())
                metrics['code_perplexity'].append(avg_code_perplexity.tolist())
                
                # Reset accumulators
                epoch_peakiness = []
                epoch_perplexity = []
                code_peakiness_sum = np.zeros(args.n_codes)
                code_perplexity_sum = np.zeros(args.n_codes)
                code_counts = 0
                
                # Print metrics
                print(f"\nStep {global_step}/{args.max_steps}, Loss: {avg_loss:.4f}, "
                      f"Recon: {avg_recon_loss:.4f}, Entropy: {avg_entropy_loss:.4f}, "
                      f"Peakiness: {avg_peakiness:.4f}, Perplexity: {avg_perplexity:.2f}")
                
                # Visualize reconstructions
                visualize_reconstructions(
                    data[:8], 
                    recon[:8], 
                    save_path=recon_dir / f"recon_step{global_step}.png"
                )
                
                # Visualize code distributions with average metrics
                visualize_code_distributions(
                    log_alpha[:2],  # Only show 2 samples max
                    temp_tensor,
                    avg_peakiness=avg_code_peakiness,
                    avg_perplexity=avg_code_perplexity,
                    save_path=dist_dir / f"dist_step{global_step}.png"
                )
                
                # Evaluate on test set
                model.eval()
                test_recon_loss = 0
                test_samples = []
                
                with torch.no_grad():
                    for i, (data, _) in enumerate(test_loader):
                        # Forward pass
                        recon, log_alpha, z, _, _ = model(data, temp_tensor)
                        
                        # Compute reconstruction loss - using mean
                        recon_loss = F.binary_cross_entropy(recon, data, reduction='mean')
                        test_recon_loss += recon_loss.item()
                        
                        # Store some samples for visualization
                        if i == 0:
                            test_samples = (data[:8], recon[:8], log_alpha[:8])
                        
                        # Limit the number of test batches to 5 for faster evaluation
                        if i >= 10:  # This means we process 5 batches (0-4)
                            break
                
                # Adjust the average calculation to account for the limited number of batches
                test_recon_loss /= min(5, len(test_loader))
                print(f"Test Reconstruction Loss: {test_recon_loss:.4f}")
                
                # Visualize test reconstructions
                visualize_reconstructions(
                    test_samples[0], 
                    test_samples[1], 
                    save_path=recon_dir / f"test_recon_step{global_step}.png"
                )
                
                # Visualize test code distributions with average metrics
                visualize_code_distributions(
                    test_samples[2][:2],  # Only show 2 samples max
                    temp_tensor,
                    avg_peakiness=avg_code_peakiness,
                    avg_perplexity=avg_code_perplexity,
                    save_path=dist_dir / f"test_dist_step{global_step}.png"
                )
                
                # Create plots for metrics over time
                create_metrics_plots(metrics, metrics_dir, args)
                
                # Switch back to training mode
                model.train()
        
        epoch += 1
    
    progress_bar.close()
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    print(f"Experiment completed. Results saved to {output_dir}")

def create_metrics_plots(metrics, output_dir, args):
    """
    Create plots showing how metrics evolve over time.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save plots
        args: Command line arguments
    """
    steps = metrics['steps']
    
    # Plot loss components
    plt.figure(figsize=(12, 6))
    plt.plot(steps, metrics['train_loss'], label='Total Loss')
    plt.plot(steps, metrics['recon_loss'], label='Reconstruction Loss')
    plt.plot(steps, metrics['entropy_loss'], label='Entropy Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Components Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_components.png")
    plt.close()
    
    # Plot temperature and entropy weight
    plt.figure(figsize=(12, 6))
    plt.plot(steps, metrics['temperature'], label='Temperature')
    plt.plot(steps, metrics['entropy_weight'], label='Entropy Weight')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Temperature and Entropy Weight Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "temp_entropy_weight.png")
    plt.close()
    
    # Plot average peakiness and perplexity
    plt.figure(figsize=(12, 6))
    plt.plot(steps, metrics['avg_peakiness'], label='Average Peakiness')
    plt.plot(steps, [p/args.codebook_size for p in metrics['avg_perplexity']], 
             label=f'Average Perplexity / {args.codebook_size}')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Average Peakiness and Normalized Perplexity Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "peakiness_perplexity.png")
    plt.close()
    
    # Plot peakiness for each code position
    plt.figure(figsize=(12, 6))
    code_peakiness = np.array(metrics['code_peakiness'])
    for i in range(args.n_codes):  # Plot all code positions
        plt.plot(steps, code_peakiness[:, i], label=f'Code {i+1}')
    plt.xlabel('Steps')
    plt.ylabel('Peakiness')
    plt.title('Peakiness by Code Position Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "code_peakiness.png")
    plt.close()
    
    # Plot perplexity for each code position
    plt.figure(figsize=(12, 6))
    code_perplexity = np.array(metrics['code_perplexity'])
    for i in range(args.n_codes):  # Plot all code positions
        plt.plot(steps, code_perplexity[:, i], label=f'Code {i+1}')
    plt.xlabel('Steps')
    plt.ylabel('Perplexity')
    plt.title('Perplexity by Code Position Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "code_perplexity.png")
    plt.close()
    
    # Create a combined plot showing reconstruction loss vs. discreteness
    plt.figure(figsize=(12, 6))
    plt.scatter(metrics['recon_loss'], metrics['avg_peakiness'], 
                c=range(len(steps)), cmap='viridis', s=50)
    plt.colorbar(label='Step')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Average Peakiness')
    plt.title('Reconstruction Loss vs. Discreteness')
    plt.grid(True)
    plt.savefig(output_dir / "recon_vs_discreteness.png")
    plt.close()

def analyze_trained_model(model_path, n_samples=10, temperature=1.0):
    """
    Analyze a trained model by:
    1. Comparing soft vs. hard discretization
    2. Visualizing the effect of changing individual codes
    
    Args:
        model_path: Path to the saved model
        n_samples: Number of samples to analyze
        temperature: Temperature for the model
    """
    # Load the model
    model_path = Path(model_path)
    checkpoint = torch.load(model_path)
    
    # Extract model parameters from the args.txt file in the same directory as the model
    model_dir = model_path.parent
    args_path = model_dir / "args.txt"
    
    # If args.txt is not in the same directory, try looking one level up
    if not args_path.exists():
        model_dir = model_path.parent.parent
        args_path = model_dir / "args.txt"
    
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find args.txt in {model_path.parent} or {model_path.parent.parent}")
    
    with open(args_path, "r") as f:
        args_dict = {}
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                args_dict[key.strip()] = value.strip()
    
    print(f"Loaded model parameters from {args_path}")
    
    # Create model with the same parameters
    model = DiscreteVAE(
        latent_dim=int(args_dict.get('latent_dim', 64)),
        codebook_size=int(args_dict.get('codebook_size', 512)),
        n_codes=int(args_dict.get('n_codes', 8)),
        hidden_dim=int(args_dict.get('hidden_dim', 400)),
        use_exp_relaxed=args_dict.get('use_exp_relaxed', 'False').lower() == 'true',
        sampling=args_dict.get('sampling', 'False').lower() == 'true'
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create output directory for analysis results
    analysis_dir = model_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    print(f"Saving analysis results to {analysis_dir}")
    
    # Load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./.data', train=False, download=True, transform=transform)
    
    # Create a smaller test loader with just enough samples
    test_loader = DataLoader(test_dataset, batch_size=n_samples, shuffle=True)
    
    # Get a batch of test images
    data, labels = next(iter(test_loader))
    
    # Set temperature
    temp_tensor = torch.tensor(temperature)
    
    # 1. Compare soft vs. hard discretization
    with torch.no_grad():
        # Standard forward pass (soft discretization)
        recon_soft, log_alpha, z_soft, latent, quantized = model(data, temp_tensor)
        
        # Hard discretization (one-hot via argmax)
        # First get the logits
        _, log_alpha, _, _, _ = model(data, temp_tensor)
        
        # Convert to one-hot via argmax
        z_hard = torch.zeros_like(z_soft)
        indices = torch.argmax(log_alpha, dim=-1)
        z_hard.scatter_(-1, indices.unsqueeze(-1), 1.0)
        
        # Get the codebook values for these one-hot vectors
        latent_pos_indices = model.latent_pos_indices.expand(data.size(0), -1)
        
        # Run through the codebook with the hard one-hot vectors
        # We need to manually compute the attention output
        normed_context = model.codebook.norm_context(model.codebook.codebook)
        normed_queries = model.codebook.norm_queries(latent)
        
        # Compute values from the codebook
        v = model.codebook.value_proj(normed_context).unsqueeze(0).expand(data.size(0), -1, -1)
        
        # Compute the attention output using the hard one-hot vectors
        attn_output_hard = torch.matmul(z_hard, v)
        y_hard = model.codebook.c_proj(attn_output_hard)
        
        # Apply the feedforward layer
        quantized_hard = y_hard + model.codebook.ff(model.codebook.norm_ff(y_hard))
        
        # Decode the hard quantized representation
        recon_hard = model.decoder(quantized_hard)
    
    # Create a figure to compare original, soft reconstruction, and hard reconstruction
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        # Original
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(data[i, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        
        # Soft reconstruction
        plt.subplot(3, n_samples, i + 1 + n_samples)
        plt.imshow(recon_soft[i, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Soft Reconstruction')
        
        # Hard reconstruction
        plt.subplot(3, n_samples, i + 1 + 2*n_samples)
        plt.imshow(recon_hard[i, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Hard Reconstruction (argmax)')
    
    plt.tight_layout()
    plt.savefig(analysis_dir / "soft_vs_hard_reconstruction.png")
    plt.close()
    
    # Calculate reconstruction error for both methods
    soft_recon_error = F.binary_cross_entropy(recon_soft, data, reduction='mean').item()
    hard_recon_error = F.binary_cross_entropy(recon_hard, data, reduction='mean').item()
    
    print(f"Soft reconstruction error: {soft_recon_error:.6f}")
    print(f"Hard reconstruction error: {hard_recon_error:.6f}")
    print(f"Difference: {hard_recon_error - soft_recon_error:.6f}")
    
    # 2. Visualize the effect of changing individual codes
    # Select a sample to modify
    sample_idx = 0  # Use the first sample
    
    # Get the original codes (argmax indices)
    original_indices = torch.argmax(log_alpha[sample_idx], dim=-1)  # [N]
    
    # For each code position, try all possible codebook entries
    n_codes = model.encoder.n_codes
    codebook_size = model.codebook.codebook_size
    
    # Print information about the codes
    print(f"Original digit: {labels[sample_idx].item()}")
    print(f"Original codes: {original_indices.tolist()}")
    
    for code_pos in range(n_codes):
        # Create a grid of images for all codebook values
        # Calculate grid dimensions - aim for roughly square layout
        grid_cols = min(16, codebook_size)  # Limit to 16 columns max for readability
        grid_rows = (codebook_size + grid_cols - 1) // grid_cols  # Ceiling division
        
        plt.figure(figsize=(grid_cols * 1.5, grid_rows * 1.5 + 1))
        
        # Show the original image at the top, spanning multiple columns
        ax_orig = plt.subplot2grid((grid_rows + 1, grid_cols), (0, 0), colspan=min(4, grid_cols))
        ax_orig.imshow(data[sample_idx, 0].detach().cpu().numpy(), cmap='gray')
        ax_orig.axis('off')
        ax_orig.set_title(f'Original Digit: {labels[sample_idx].item()}\nCode {code_pos+1} Original Value: {original_indices[code_pos].item()}')
        
        # For each variation, change one code and show the result
        for var_idx in range(codebook_size):
            # Create a copy of the original indices
            modified_indices = original_indices.clone()
            
            # Change the code at the specified position
            modified_indices[code_pos] = var_idx
            
            # Create one-hot vectors from these indices
            z_modified = torch.zeros_like(z_soft[sample_idx:sample_idx+1])
            z_modified.scatter_(-1, modified_indices.unsqueeze(0).unsqueeze(-1), 1.0)
            
            # Get the codebook values for these one-hot vectors
            with torch.no_grad():
                # Compute values from the codebook
                v = model.codebook.value_proj(normed_context).unsqueeze(0)
                
                # Compute the attention output using the modified one-hot vectors
                attn_output_mod = torch.matmul(z_modified, v)
                y_mod = model.codebook.c_proj(attn_output_mod)
                
                # Apply the feedforward layer
                quantized_mod = y_mod + model.codebook.ff(model.codebook.norm_ff(y_mod))
                
                # Decode the modified quantized representation
                recon_mod = model.decoder(quantized_mod)
            
            # Calculate row and column in the grid
            row = (var_idx // grid_cols) + 1  # +1 because original image is in row 0
            col = var_idx % grid_cols
            
            # Show the modified reconstruction
            ax = plt.subplot2grid((grid_rows + 1, grid_cols), (row, col))
            ax.imshow(recon_mod[0, 0].detach().cpu().numpy(), cmap='gray')
            ax.axis('off')
            
            # Highlight the original value with a colored border
            if var_idx == original_indices[code_pos].item():
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('red')
                    spine.set_linewidth(3)
            
            ax.set_title(f'{var_idx}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(analysis_dir / f"code_{code_pos+1}_all_variations.png")
        plt.close()
        
        # Also create a more focused visualization with just a subset of variations
        # This is helpful when codebook size is very large
        n_variations = min(10, codebook_size)
        plt.figure(figsize=(15, 3))
        
        # Show the original image
        plt.subplot(1, n_variations + 1, 1)
        plt.imshow(data[sample_idx, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'Original\nDigit: {labels[sample_idx].item()}')
        
        # Include the original value and some values around it
        original_value = original_indices[code_pos].item()
        
        # Generate a list of indices to visualize, centered around the original value
        if codebook_size <= n_variations:
            # If codebook is small enough, show all values
            indices_to_show = list(range(codebook_size))
        else:
            # Otherwise, show the original value and some values around it
            half_range = (n_variations - 1) // 2
            start_idx = max(0, original_value - half_range)
            end_idx = min(codebook_size, start_idx + n_variations)
            
            # Adjust start if we hit the upper bound
            if end_idx == codebook_size:
                start_idx = max(0, codebook_size - n_variations)
                
            indices_to_show = list(range(start_idx, end_idx))
        
        # For each selected variation, show the result
        for i, var_idx in enumerate(indices_to_show):
            # Create a copy of the original indices
            modified_indices = original_indices.clone()
            
            # Change the code at the specified position
            modified_indices[code_pos] = var_idx
            
            # Create one-hot vectors from these indices
            z_modified = torch.zeros_like(z_soft[sample_idx:sample_idx+1])
            z_modified.scatter_(-1, modified_indices.unsqueeze(0).unsqueeze(-1), 1.0)
            
            # Get the codebook values for these one-hot vectors
            with torch.no_grad():
                # Compute values from the codebook
                v = model.codebook.value_proj(normed_context).unsqueeze(0)
                
                # Compute the attention output using the modified one-hot vectors
                attn_output_mod = torch.matmul(z_modified, v)
                y_mod = model.codebook.c_proj(attn_output_mod)
                
                # Apply the feedforward layer
                quantized_mod = y_mod + model.codebook.ff(model.codebook.norm_ff(y_mod))
                
                # Decode the modified quantized representation
                recon_mod = model.decoder(quantized_mod)
            
            # Show the modified reconstruction
            plt.subplot(1, len(indices_to_show) + 1, i + 2)
            plt.imshow(recon_mod[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            
            # Highlight the original value
            title = f'Code {code_pos+1}\nValue: {var_idx}'
            if var_idx == original_value:
                title += ' (orig)'
                plt.gca().spines['bottom'].set_color('red')
                plt.gca().spines['top'].set_color('red')
                plt.gca().spines['right'].set_color('red')
                plt.gca().spines['left'].set_color('red')
                plt.gca().spines['bottom'].set_visible(True)
                plt.gca().spines['top'].set_visible(True)
                plt.gca().spines['right'].set_visible(True)
                plt.gca().spines['left'].set_visible(True)
                plt.gca().spines['bottom'].set_linewidth(2)
                plt.gca().spines['top'].set_linewidth(2)
                plt.gca().spines['right'].set_linewidth(2)
                plt.gca().spines['left'].set_linewidth(2)
            
            plt.title(title)
        
        plt.tight_layout()
        plt.savefig(analysis_dir / f"code_{code_pos+1}_focused_variations.png")
        plt.close()
    
    print(f"Analysis complete. Results saved to {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description='Study discrete representation learning on MNIST')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension (D)')
    parser.add_argument('--n_codes', type=int, default=8, help='Number of discrete codes (N)')
    parser.add_argument('--hidden_dim', type=int, default=400, help='Hidden dimension')
    parser.add_argument('--codebook_size', type=int, default=512, help='Codebook size')
    parser.add_argument('--use_exp_relaxed', action='store_true', help='Use exponentially relaxed distribution')
    parser.add_argument('--sampling', action='store_true', help='Use sampling instead of softmax')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of training steps')
    parser.add_argument('--val_check_interval', type=int, default=100, help='Validation check interval (in steps)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Temperature parameters
    parser.add_argument('--min_temp', type=float, default=1.0, help='Minimum temperature')
    parser.add_argument('--max_temp', type=float, default=1.0, help='Maximum temperature')
    parser.add_argument('--temp_anneal_steps', type=int, default=0, help='Number of steps for temperature annealing (0 = no annealing)')
    
    # Entropy weight parameters
    parser.add_argument('--min_entropy_weight', type=float, default=0.0, help='Minimum entropy weight')
    parser.add_argument('--max_entropy_weight', type=float, default=0.0, help='Maximum entropy weight')
    parser.add_argument('--entropy_anneal_steps', type=int, default=0, help='Number of steps for entropy weight annealing (0 = no annealing)')
    
    # Diversity parameters
    parser.add_argument('--diversity_weight', type=float, default=0.1, help='Weight for the codebook diversity loss')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='mnist_discrete_vae', help='Output directory')
    
    # Analysis parameters
    parser.add_argument('--analyze', action='store_true', help='Run analysis on a trained model')
    parser.add_argument('--model_path', type=str, help='Path to the trained model for analysis')
    parser.add_argument('--analysis_samples', type=int, default=10, help='Number of samples to use for analysis')
    parser.add_argument('--analysis_temp', type=float, default=1.0, help='Temperature to use for analysis')
    
    args = parser.parse_args()
    
    if args.analyze and args.model_path:
        analyze_trained_model(args.model_path, args.analysis_samples, args.analysis_temp)
    else:
        run_experiment(args)

if __name__ == "__main__":
    main()