import torch
import torch.nn.functional as F

def vectorized_point_distance(points, ref_point):
    """
    Computes the distance between each point in `points` and a reference point.
    The distance is defined as the sum (over S) of the Euclidean norms computed
    on the D dimension of each corresponding latent vector.
    
    Args:
        points: Tensor of shape [M, B, S, D] (M points in trajectory).
        ref_point: Tensor of shape [B, S, D].
        
    Returns:
        distances: Tensor of shape [M, B].
    """
    # Expand ref_point to [1, B, S, D] and subtract; result: [M, B, S, D]
    diff = points - ref_point.unsqueeze(0)
    # Compute the L2 norm over the last dimension (D) -> shape [M, B, S]
    norms = torch.norm(diff, dim=-1)
    # Sum over the S dimension -> shape [M, B]
    distances = norms.sum(dim=2)
    return distances

def vectorized_monotonic_trajectory_loss(traj, src, dest, margin=0.0):
    """
    Computes a loss that enforces the following:
      1. For the source side, each successive point is at least `margin` further away from src.
      2. For the destination side, each successive point is at least `margin` closer to dest,
         except for the final transition (last intermediate -> dest) where margin is set to 0.
    
    Here the distance between two points (of shape [B, S, D]) is defined as the sum over S
    of the Euclidean distances between the corresponding D-dimensional vectors.
    
    Args:
        traj: Tensor of shape [N, B, S, D] representing the intermediate trajectory points.
        src: Tensor of shape [B, S, D] representing the source.
        dest: Tensor of shape [B, S, D] representing the destination.
        margin: A non-negative float margin to enforce for all adjacent pairs on the source side
                and for the destination side except the final adjacent pair.
                
    Returns:
        A scalar loss value averaged over adjacent pairs and the batch.
    """
    # Form the full trajectory by concatenating src at the beginning and dest at the end.
    # full_traj becomes a tensor of shape [M, B, S, D] with M = N + 2.
    full_traj = torch.cat([src.unsqueeze(0), traj, dest.unsqueeze(0)], dim=0)
    M = full_traj.shape[0]

    # Compute distances between each point in the full trajectory and src.
    d_src = vectorized_point_distance(full_traj, src)   # Shape: [M, B]
    # Compute distances between each point in the full trajectory and dest.
    d_dest = vectorized_point_distance(full_traj, dest)  # Shape: [M, B]

    # Compute the source-side differences.
    # We want: d_src[i] + margin <= d_src[i+1]  --> d_src[i] - d_src[i+1] + margin <= 0.
    diff_src = d_src[:-1] - d_src[1:] + margin  # Shape: [M-1, B]
    loss_src = F.relu(diff_src).mean()

    # Compute the destination-side differences.
    # We want: d_dest[i+1] + margin <= d_dest[i]  --> d_dest[i+1] - d_dest[i] + margin <= 0.
    diff_dest = d_dest[1:] - d_dest[:-1] + margin  # Shape: [M-1, B]
    # Build a margin vector with shape [M-1, B] that applies the margin to every adjacent pair except the last:
    loss_dest = F.relu(diff_dest).mean()

    total_loss = loss_src + loss_dest
    return total_loss

# Example usage:
if __name__ == "__main__":
    B, S, D = 8, 10, 32   # Batch size, number of latent vectors per point, and latent dimension.
    N = 5                 # Number of intermediate trajectory points.
    
    # Dummy tensors for source and destination (B x S x D).
    src = torch.randn(B, S, D)
    dest = torch.randn(B, S, D)
    # Dummy tensor for intermediate trajectory points (N x B x S x D).
    traj = torch.randn(N, B, S, D)
    
    loss = vectorized_monotonic_trajectory_loss(traj, src, dest, margin=0.1)
    print("Vectorized Monotonic Trajectory Loss:", loss.item())
