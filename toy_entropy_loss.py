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

def entropy_loss(log_alpha, reduction="mean"):
    """
    Compute the entropy of the latent distribution.
    
    Args:
        log_alpha: Logits for the latent distribution, shape [B, codebook_size]
        reduction: Reduction method ('sum', 'mean', or 'batchmean')
                
    Returns:
        Entropy reduced according to the specified method
    """
    # Compute log probabilities using log_softmax
    log_probs = F.log_softmax(log_alpha, dim=-1)
    
    # Get probabilities by exponentiating log probabilities
    probs = torch.exp(log_probs)
    
    # Compute entropy: -sum(p * log(p))
    entropy_per_sample = -torch.sum(probs * log_probs, dim=-1)  # [B]
    
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

def compute_perplexity(probs):
    """
    Compute perplexity of probability distributions.
    Perplexity = 2^(entropy), measures how uniform the distribution is.
    
    Args:
        probs: Probability distributions [B, N, codebook_size]
        
    Returns:
        Perplexity per sample [B, N]
    """
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log2(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, N]
    
    # Perplexity = 2^(entropy)
    perplexity = 2.0 ** entropy
    
    return perplexity

def compute_peakiness(probs):
    """
    Compute peakiness (max probability) of distributions.
    
    Args:
        probs: Probability distributions [B, N, codebook_size]
        
    Returns:
        Peakiness per sample [B, N]
    """
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

def visualize_code_distributions(probs, avg_peakiness=None, avg_perplexity=None, save_path=None):
    """
    Visualize probability distributions for each code position.
    
    Args:
        probs: Probability distributions [B, N, codebook_size]
        avg_peakiness: Average peakiness for each code position [N] (optional)
        avg_perplexity: Average perplexity for each code position [N] (optional)
        save_path: Path to save the figure (optional)
    """
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
            
            # Plot distribution
            ax.bar(range(len(probs_to_plot)), probs_to_plot)
            ax.set_ylim(0, max_prob * 1.05)  # Set y-axis limit to global max with 5% padding
            
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
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, log_alpha, z, _, _ = model(data, torch.tensor(temperature))
            
            # Compute losses - using mean reconstruction loss
            recon_loss = F.binary_cross_entropy(recon, data, reduction='mean')
            ent_loss = entropy_loss(log_alpha, reduction="mean")
            
            # Total loss - ADDING entropy loss to encourage discrete selection
            loss = recon_loss + entropy_weight * ent_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Compute probabilities, peakiness, and perplexity
            probs = torch.softmax(log_alpha, dim=-1)  # [B, N, codebook_size]
            
            # Calculate peakiness and perplexity for each code position
            batch_peakiness = compute_peakiness(probs).detach().cpu().numpy()  # [B, N]
            batch_perplexity = compute_perplexity(probs).detach().cpu().numpy()  # [B, N]
            
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
                    probs[:2],  # Only show 2 samples max
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
                        recon, log_alpha, z, _, _ = model(data, torch.tensor(temperature))
                        
                        # Compute reconstruction loss - using mean
                        recon_loss = F.binary_cross_entropy(recon, data, reduction='mean')
                        test_recon_loss += recon_loss.item()
                        
                        # Store some samples for visualization
                        if i == 0:
                            test_samples = (data[:8], recon[:8], torch.softmax(log_alpha[:8], dim=-1))
                
                test_recon_loss /= len(test_loader)
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
    parser.add_argument('--val_check_interval', type=int, default=500, help='Validation check interval (in steps)')
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
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='mnist_discrete_vae', help='Output directory')
    
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()