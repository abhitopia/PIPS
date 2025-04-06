#%%
%matplotlib inline

#%%
import numpy as np

def find_intermediate_nodes(codebook, idx1, idx2):
    """
    Given a codebook (C x D numpy array) and two indices (idx1, idx2) representing 
    endpoints in the codebook, return a list of intermediate node indices that satisfy:
    
    1. dist(idx1, n) < dist(idx1, idx2)
    2. dist(n, idx2) < dist(idx1, idx2)
    3. (Optionally) dist(idx1, n) + dist(n, idx2) > dist(idx1, idx2)
    
    The returned list is sorted in increasing order of distance from idx1.
    """
    vec1 = codebook[idx1]
    vec2 = codebook[idx2]
    
    # Euclidean distance between the endpoints
    dist_12 = np.linalg.norm(vec1 - vec2)
    
    intermediate_nodes = []
    
    # Loop over all potential intermediate nodes
    for n in range(codebook.shape[0]):
        if n == idx1 or n == idx2:
            continue
        
        vec_n = codebook[n]
        d1 = np.linalg.norm(vec1 - vec_n)
        d2 = np.linalg.norm(vec2 - vec_n)
        
        # Check if the candidate node is "between" the endpoints
        if d1 < dist_12 and d2 < dist_12 and (d1 + d2 > dist_12):
            intermediate_nodes.append((n, d1))
    
    # Sort nodes by distance from the first endpoint (idx1)
    intermediate_nodes.sort(key=lambda x: x[1])
    
    # Return only the indices, in order
    return [n for n, d in intermediate_nodes]


from tqdm import tqdm

def precompute_intermediate_nodes(codebook):
    """
    Precompute a dictionary mapping each pair (i, j) of distinct codebook indices 
    to the list of intermediate node indices that lie between them.
    
    This structure can be used later to guide beam search across the discrete latent space.
    """
    C = codebook.shape[0]
    intermediate_dict = {}
    
    progress_bar = tqdm(total=C * C, desc="Precomputing distances")

    for i in range(C):
        for j in range(C):

            idx = i * C + j
            progress_bar.update(1)
            if i == j:
                continue
            intermediate_dict[(i, j)] = find_intermediate_nodes(codebook, i, j)
    
    return intermediate_dict

#%%
from pathlib import Path
from pips.misc.artifact import Artifact
import wandb
import torch
from train_vqvae import ExperimentConfig, VQVAE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg', 'nbAgg' for Jupyter
from IPython.display import display

project_name = "DLR_v3"
model_src = "v5_rope_ape_d512_quant_v3/best/30000"
checkpoint_dir = "runs"

# Parse model source string first
source_project, run_name, category, alias = Artifact.parse_artifact_string(
    model_src,
    default_project=project_name
)

# Initialize artifact manager with correct project and run name
artifact_manager = Artifact(
    entity=wandb.api.default_entity,
    project_name=source_project,
    run_name=run_name
)

# Get local checkpoint path
checkpoint_path = artifact_manager.get_local_checkpoint(
    category=category,
    alias=alias,
    checkpoint_dir=Path(checkpoint_dir)
)

ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model_config = ckpt['hyper_parameters']['experiment_config'].model_config
sd = {k.replace('_orig_mod.', '').replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
model_config.use_ema = True
model = VQVAE(model_config)
model.load_state_dict(sd)# %%

# %%
def visualize_grid_comparison(input_grid, recon_grid=None, titles=None, save_path=None):
    """
    Visualizes one or two grids side by side with the same colormap.
    
    Args:
        input_grid: Input grid as a 2D numpy array
        recon_grid: Optional reconstruction grid as a 2D numpy array
        titles: Optional list of titles for the plots
        save_path: Optional path to save the figure
    """
    # Ensure input is 2D
    if input_grid.ndim > 2:
        if input_grid.ndim == 1:
            size = int(np.sqrt(input_grid.shape[0]))
            input_grid = input_grid.reshape(size, size)
        else:
            input_grid = input_grid[0]
    
    if recon_grid is not None and recon_grid.ndim > 2:
        if recon_grid.ndim == 1:
            size = int(np.sqrt(recon_grid.shape[0]))
            recon_grid = recon_grid.reshape(size, size)
        else:
            recon_grid = recon_grid[0]
    
    # Define specific colors for values 0-9 and 15
    color_map = {
        0: '#000000',  # black
        1: '#0074D9',  # blue
        2: '#FF4136',  # red
        3: '#2ECC40',  # green
        4: '#FFDC00',  # yellow
        5: '#AAAAAA',  # grey
        6: '#F012BE',  # fuschia
        7: '#FF851B',  # orange
        8: '#7FDBFF',  # teal
        9: '#870C25',  # brown
        15: '#FFFFFF'  # white
    }
    
    # Default color for other values
    default_color = '#FF00FF'
    
    # Get all unique values
    if recon_grid is not None:
        all_values = np.unique(np.concatenate([input_grid.flatten(), recon_grid.flatten()]))
    else:
        all_values = np.unique(input_grid)
    
    # Create custom colormap
    colors = []
    for val in range(max(all_values.max() + 1, 16)):  # Ensure at least 16 colors (for value 15)
        if val in color_map:
            colors.append(color_map[val])
        else:
            colors.append(default_color)
    
    custom_cmap = mcolors.ListedColormap(colors)
    
    if recon_grid is not None:
        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(18, 7))
        
        # Create a gridspec layout with space for the colorbar
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.1])
        
        # Create separate axes for input, reconstruction, and colorbar
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cbar_ax = fig.add_subplot(gs[0, 2])
        
        # Default titles
        if titles is None:
            titles = ["Input Grid", "Reconstruction"]
        
        # Plot input grid
        im1 = ax1.imshow(input_grid, cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
        ax1.set_title(titles[0])
        ax1.set_xticks(np.arange(-0.5, input_grid.shape[1], 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, input_grid.shape[0], 1), minor=True)
        ax1.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
        
        # Plot reconstruction grid
        im2 = ax2.imshow(recon_grid, cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
        ax2.set_title(titles[1])
        ax2.set_xticks(np.arange(-0.5, recon_grid.shape[1], 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, recon_grid.shape[0], 1), minor=True)
        ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
        
        # Add colorbar in its own dedicated axis
        cbar = plt.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Grid Values')
        
        # Add ticks for the values that appear in the data
        present_values = sorted(all_values)
        cbar.set_ticks(present_values)
        cbar.set_ticklabels([str(int(v)) for v in present_values])
    else:
        # If no reconstruction, just show input
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Default title
        if titles is None:
            titles = ["Input Grid"]
        
        # Plot the grid
        im = ax.imshow(input_grid, cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
        ax.set_title(titles[0])
        ax.set_xticks(np.arange(-0.5, input_grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, input_grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Grid Values')
        
        # Add ticks for the values that appear in the data
        present_values = sorted(np.unique(input_grid))
        cbar.set_ticks(present_values)
        cbar.set_ticklabels([str(int(v)) for v in present_values])
    
    plt.subplots_adjust(wspace=0.05)
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        return fig


def visualize_indices_comparison(indices_1, indices_2, codebook_size=512):
    """
    Creates three simple visualizations of codebook indices:
    1. First grid shows indices from indices_1
    2. Second grid shows indices from indices_2
    3. Third grid shows comparison (black=same, red=only in first, green=only in second)
    
    Args:
        indices_1: First set of indices [batch_size, num_positions]
        indices_2: Second set of indices [batch_size, num_positions]
        codebook_size: Size of the codebook (default 512)
    """
    # Ensure indices are numpy arrays and take first batch item if needed
    if isinstance(indices_1, torch.Tensor):
        indices_1 = indices_1.cpu().numpy()
    if isinstance(indices_2, torch.Tensor):
        indices_2 = indices_2.cpu().numpy()
    
    if indices_1.ndim > 1:
        indices_1 = indices_1[0]
    if indices_2.ndim > 1:
        indices_2 = indices_2[0]
    
    # Number of positions
    num_positions = indices_1.shape[0]
    
    # Create the binary grids (positions × codebook_size)
    grid_1 = np.zeros((num_positions, codebook_size))
    grid_2 = np.zeros((num_positions, codebook_size))
    
    # Fill in the grids - mark each selected index with a 1
    for pos in range(num_positions):
        idx_1 = indices_1[pos]
        idx_2 = indices_2[pos]
        
        if idx_1 < codebook_size:
            grid_1[pos, idx_1] = 1
            
        if idx_2 < codebook_size:
            grid_2[pos, idx_2] = 1
    
    # Create the comparison grid
    # 0: same in both (black)
    # 1: only in first (red)
    # 2: only in second (green)
    grid_compare = np.zeros((num_positions, codebook_size))
    
    for pos in range(num_positions):
        idx_1 = indices_1[pos]
        idx_2 = indices_2[pos]
        
        if idx_1 == idx_2:
            # Same index used in both
            if idx_1 < codebook_size:
                grid_compare[pos, idx_1] = 1  # Black (will use custom colormap)
        else:
            # Different indices used
            if idx_1 < codebook_size:
                grid_compare[pos, idx_1] = 2  # Red
            if idx_2 < codebook_size:
                grid_compare[pos, idx_2] = 3  # Green
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot first grid
    axes[0].imshow(grid_1, cmap='Blues', aspect='auto')
    axes[0].set_title('Indices from Set 1')
    axes[0].set_xlabel('Codebook Index')
    axes[0].set_ylabel('Position')
    
    # Plot second grid
    axes[1].imshow(grid_2, cmap='Blues', aspect='auto')
    axes[1].set_title('Indices from Set 2')
    axes[1].set_xlabel('Codebook Index')
    axes[1].set_ylabel('Position')
    
    # Custom colormap for comparison grid
    # 0: not used (white)
    # 1: same in both (black)
    # 2: only in first (red)
    # 3: only in second (green)
    colors = ['white', 'black', 'red', 'green']
    custom_cmap = mcolors.ListedColormap(colors)
    
    # Plot comparison grid
    im = axes[2].imshow(grid_compare, cmap=custom_cmap, vmin=0, vmax=3, aspect='auto')
    axes[2].set_title('Comparison (black=same, red=only in set 1, green=only in set 2)')
    axes[2].set_xlabel('Codebook Index')
    axes[2].set_ylabel('Position')
    
    # Add a colorbar for the comparison
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.2])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(['Not used', 'Same in both', 'Only in set 1', 'Only in set 2'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
    return fig

#%%
from functools import partial
from pips.grid_dataset import GridDataset, DatasetType
val_dataset = GridDataset(dataset_type=DatasetType.VAL)

collate_fn_val = partial(
    GridDataset.collate_fn_project,
    pad_value=model_config.padding_idx,
    permute=False,
    max_height=model_config.max_grid_height,
    max_width=model_config.max_grid_width
    )

def run_model(idx):
    x = collate_fn_val([val_dataset[idx]])[0]
    print(val_dataset[idx])
    model.eval()

    with torch.no_grad():
        logits, losses, indices = model(x)

    input_grid = x.numpy()[0].reshape(32, 32)  # Reshape and take first sample
    recon_grid = logits.argmax(dim=-1).numpy()[0].reshape(32, 32)  # Model output
    fig = visualize_grid_comparison(input_grid, recon_grid)

    return input_grid, recon_grid, fig, indices

def decode_indices(indices):
    grid_pos_indices = model.grid_pos_indices.expand(1, -1, -1)
    latent_pos_indices = model.latent_pos_indices.expand(1, -1)
    decoder_input = torch.index_select(model.codebook.vq_embs.weight, dim=0, index=indices.flatten()).unsqueeze(0)
    logits = model.decode(decoder_input, grid_pos_indices, latent_pos_indices)[0]
    recon_grid = logits.argmax(dim=-1).numpy()[0].reshape(32, 32)  # Model output
    fig = visualize_grid_comparison(recon_grid, None)
    display(fig)

input_grid_1, recon_grid_1, fig_1, indices_1 = run_model(72)
input_grid_2, recon_grid_2, fig_2, indices_2 = run_model(73)

#%%
display(fig_1)
display(fig_2)
# %%
for idx, (c1, c2) in enumerate(zip(indices_1.flatten(), indices_2.flatten())):
    c1 = c1.item()
    c2 = c2.item()
    if c1 == c2:
        print(f"{idx}: {c1}")
    if c1 != c2:
        print(f"{idx}: {c1}->{c2}")
#%%
# intemediate_dict = precompute_intermediate_nodes(model.codebook.vq_embs.weight.detach().numpy())
import pickle
with open("intemediate_dict.pkl", "rb") as f:
    intemediate_dict = pickle.load(f)
# %%
# import pickle
# with open("intemediate_dict.pkl", "wb") as f:
#     pickle.dump(intemediate_dict, f)
# %%
def cost_function(state, target, codebook):
    """
    Compute a global cost for a candidate state relative to the target state.
    Here the cost is defined as the sum (over all dimensions) of the Euclidean 
    distances between the codebook entry for that dimension in the state and 
    the corresponding codebook entry in the target.
    """
    return sum(np.linalg.norm(codebook[state[i]] - codebook[target[i]]) 
               for i in range(len(state)))

def beam_search_trajectories(x1, x2, codebook, intermediate_dict, beam_width=3, max_steps=20):
    """
    Use beam search to find top candidate trajectories from the starting encoding x1 to 
    the target encoding x2, using the precomputed intermediate nodes.

    Args:
      x1: starting encoding (list/tuple of indices; length = number of latent dimensions)
      x2: target encoding (list/tuple of indices)
      codebook: a numpy array of shape (C, D)
      intermediate_dict: dictionary mapping (a, b) -> sorted list of intermediate node indices 
                         for moving from codebook index a to codebook index b
      beam_width: number of candidate trajectories to keep at each step
      max_steps: maximum number of expansion steps
      
    Returns:
      A list of trajectories. Each trajectory is a list of candidate states (tuples of indices) 
      that starts with x1 and ends with x2.
    """
    start = tuple(x1)
    target = tuple(x2)
    initial_candidate = {
        'state': start,
        'path': [start],
        'cost': cost_function(start, target, codebook)
    }
    beam = [initial_candidate]
    completed = []
    step = 0

    while beam and step < max_steps:
        next_beam = []
        for candidate in beam:
            state = candidate['state']
            # If this candidate has reached the target, keep it in completed.
            if state == target:
                completed.append(candidate)
                continue
            # For every dimension where the candidate doesn't match the target, expand moves.
            for j in range(len(state)):
                if state[j] == target[j]:
                    continue
                current_val = state[j]
                target_val = target[j]
                # Retrieve allowed moves for this dimension from the precomputed dict.
                allowed_moves = intermediate_dict.get((current_val, target_val), []).copy()
                # Also include a direct jump to the target index if not already present.
                if target_val not in allowed_moves:
                    allowed_moves.append(target_val)
                # For each allowed move, update that dimension and form a new candidate state.
                for move in allowed_moves[:1]:
                    new_state = list(state)
                    new_state[j] = move
                    new_state = tuple(new_state)
                    new_cost = cost_function(new_state, target, codebook)
                    new_candidate = {
                        'state': new_state,
                        'path': candidate['path'] + [new_state],
                        'cost': new_cost
                    }
                    next_beam.append(new_candidate)
        if not next_beam:
            break
        # Sort all new candidate states by their global cost.
        next_beam.sort(key=lambda cand: cand['cost'])
        # Keep only the top-k candidates for the next step.
        beam = next_beam[:beam_width]
        step += 1

    # Append any remaining candidates that have reached the target.
    completed.extend([cand for cand in beam if cand['state'] == target])
    # Sort completed trajectories by their cost.
    completed.sort(key=lambda cand: cand['cost'])
    trajectories = [cand['path'] for cand in completed]
    return trajectories

codebook = model.codebook.vq_embs.weight.detach().numpy()
result = beam_search_trajectories(indices_1.tolist()[0], indices_2.tolist()[0], codebook, intemediate_dict, beam_width=5, max_steps=1000)

def print_trajectory_changes(trajectory):
    """
    Given a trajectory (list of states, each a tuple of indices), print the changes between consecutive states.
    
    For each step, this function prints which dimensions changed and the transition from the old index to the new index.
    """
    if not trajectory:
        print("Empty trajectory.")
        return

    print("Trajectory changes:")
    for i in range(1, len(trajectory)):
        prev_state = trajectory[i - 1]
        curr_state = trajectory[i]
        changes = []
        for dim, (prev, curr) in enumerate(zip(prev_state, curr_state)):
            if prev != curr:
                changes.append(f"Dim {dim}: {prev} -> {curr}")
        print(f"Step {i}: " + "; ".join(changes))

print_trajectory_changes(result[0])
#%%

for indices in result[0]:
    torch_indices = torch.tensor(indices).unsqueeze(0)
    decode_indices(torch_indices)
# %%
# %%
