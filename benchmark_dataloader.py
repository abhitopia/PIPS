import time
from functools import partial
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pips.grid_dataset import TRAIN_GRID_LOADERS, VAL_GRID_LOADERS, GridDataset, load_grid_loaders
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path

# Add this at the top level of the file, outside any function
def worker_init_fn(worker_id):
    """Initialize worker for DataLoader"""
    print(f"Initializing worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._initialize_data()  # Initialize data for this worker
        print(f"Worker {worker_id} dataset size: {len(dataset)}")

def benchmark_dataloader(
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int | None = 2,
    pin_memory: bool = True,
    num_batches: int = 100,
    init_timeout: float = 30.0,
) -> Dict[str, float]:
    """
    Benchmark a specific DataLoader configuration.
    
    Returns:
        Dict containing timing metrics including initialization overhead
    """
    # Set up dataset and collate function
    project_size = (32, 32)  # Example size, adjust as needed
    padding_idx = 999  # Example value, adjust as needed
    
    print(f"\nStarting benchmark with:")
    print(f"- batch_size: {batch_size}")
    print(f"- num_workers: {num_workers}")
    print(f"- persistent_workers: {persistent_workers}")
    print(f"- prefetch_factor: {prefetch_factor}")
    print(f"- pin_memory: {pin_memory}")
    
    collate_fn = partial(
        GridDataset.collate_fn, 
        pad_value=padding_idx, 
        permute=True, 
        project_size=project_size
    )
    
    print("Creating dataset...")
    dataset = GridDataset(train=True)
    print(f"Dataset created, size: {len(dataset)}")
    print(f"Dataset type: {type(dataset)}")
    
    # Measure initialization time
    print("Creating DataLoader...")
    init_start = time.perf_counter()
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory and torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            timeout=120 if num_workers > 0 else 0
        )
        print("DataLoader created")
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        raise
    
    # Measure time to first batch (includes worker startup)
    first_batch_start = time.perf_counter()
    init_overhead = first_batch_start - init_start
    print(f"DataLoader init took: {init_overhead*1000:.2f}ms")
    
    # Get first batch to measure startup overhead
    print("Getting first batch...")
    try:
        start = time.perf_counter()
        for batch in dataloader:
            first_batch_end = time.perf_counter()
            break
        
        startup_overhead = first_batch_end - first_batch_start
        print(f"First batch retrieved in: {startup_overhead*1000:.2f}ms")
        
        if time.perf_counter() - init_start > init_timeout:
            print("Warning: Initialization took longer than timeout!")
            return {
                'init_overhead': init_overhead,
                'startup_overhead': startup_overhead,
                'total_init_time': time.perf_counter() - init_start,
                'error': 'Initialization timeout'
            }
            
    except Exception as e:
        print(f"Error getting first batch: {e}")
        return {
            'init_overhead': init_overhead,
            'startup_overhead': time.perf_counter() - first_batch_start,
            'total_init_time': time.perf_counter() - init_start,
            'error': str(e)
        }
    
    total_init_time = first_batch_end - init_start
    
    # Reset dataloader for actual benchmark if not using persistent workers
    if not persistent_workers or num_workers == 0:
        print("Recreating DataLoader for benchmark...")
        del dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory and torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            timeout=120 if num_workers > 0 else 0
        )
    
    # Timing measurements for regular batches
    print("Starting batch timing measurements...")
    start_time = time.perf_counter()
    batch_times = []
    
    try:
        for i, (x, _) in enumerate(dataloader):
            if i == 0:
                start_time = time.perf_counter()
                continue
                
            if i > num_batches:
                break
                
            batch_end = time.perf_counter()
            batch_times.append(batch_end - start_time)
            start_time = batch_end
            
            # If using GPU, include data transfer time in measurement
            if torch.cuda.is_available():
                x = x.cuda()
                torch.cuda.synchronize()
    except Exception as e:
        print(f"Error during batch processing: {e}")
        if batch_times:
            print(f"Completed {len(batch_times)} batches before error")
        else:
            return {
                'init_overhead': init_overhead,
                'startup_overhead': startup_overhead,
                'total_init_time': total_init_time,
                'error': str(e)
            }
    
    # Calculate metrics
    if batch_times:
        batch_times = np.array(batch_times)
        return {
            'init_overhead': init_overhead,
            'startup_overhead': startup_overhead,
            'total_init_time': total_init_time,
            'mean_time': np.mean(batch_times),
            'std_time': np.std(batch_times),
            'median_time': np.median(batch_times),
            'min_time': np.min(batch_times),
            'max_time': np.max(batch_times),
            'batches_per_sec': 1.0 / np.mean(batch_times),
            'samples_per_sec': batch_size / np.mean(batch_times)
        }
    else:
        return {
            'init_overhead': init_overhead,
            'startup_overhead': startup_overhead,
            'total_init_time': total_init_time,
            'error': 'No batches completed'
        }

def run_benchmarks(
    batch_sizes: List[int] = [32, 64, 128],
    num_workers_list: List[int] = [0, 1, 2, 4, 8],
    persistent_workers_list: List[bool] = [False, True],
    num_batches: int = 100
) -> pd.DataFrame:
    """
    Run benchmarks across different configurations and return results as a DataFrame.
    """
    results = []
    
    total_configs = len(batch_sizes) * len(num_workers_list) * len(persistent_workers_list)
    config_count = 0
    
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            for persistent_workers in persistent_workers_list:
                config_count += 1
                print(f"\nTesting configuration {config_count}/{total_configs}:")
                print(f"batch_size={batch_size}, num_workers={num_workers}, "
                      f"persistent_workers={persistent_workers}")
                
                # Skip invalid configurations
                if num_workers == 0 and persistent_workers:
                    print("Skipping: persistent_workers=True invalid with num_workers=0")
                    continue
                
                try:
                    metrics = benchmark_dataloader(
                        batch_size=batch_size,
                        num_workers=num_workers,
                        persistent_workers=persistent_workers,
                        num_batches=num_batches
                    )
                    
                    results.append({
                        'batch_size': batch_size,
                        'num_workers': num_workers,
                        'persistent_workers': persistent_workers,
                        **metrics
                    })
                    
                    print(f"Samples/sec: {metrics['samples_per_sec']:.2f}")
                    
                except Exception as e:
                    print(f"Error with configuration: {e}")
                    continue
    
    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate plots from benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Samples/sec vs num_workers for different batch sizes
    plt.figure(figsize=(12, 6))
    for batch_size in df['batch_size'].unique():
        for persistent in df['persistent_workers'].unique():
            data = df[(df['batch_size'] == batch_size) & 
                     (df['persistent_workers'] == persistent)]
            if not data.empty:
                label = f'batch={batch_size}, persistent={persistent}'
                plt.plot(data['num_workers'], data['samples_per_sec'], 
                        marker='o', label=label)
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Samples per Second')
    plt.title('DataLoader Performance vs Number of Workers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'workers_vs_throughput.png')
    plt.close()
    
    # Plot 2: Distribution of batch times
    plt.figure(figsize=(12, 6))
    for batch_size in df['batch_size'].unique():
        data = df[df['batch_size'] == batch_size]
        plt.errorbar(data['num_workers'], data['mean_time'], 
                    yerr=data['std_time'], label=f'batch={batch_size}',
                    marker='o', capsize=5)
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Batch Time (seconds)')
    plt.title('Batch Time Distribution vs Number of Workers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_time_distribution.png')
    plt.close()
    
    # New plot: Initialization overhead
    plt.figure(figsize=(12, 6))
    for batch_size in df['batch_size'].unique():
        for persistent in df['persistent_workers'].unique():
            data = df[(df['batch_size'] == batch_size) & 
                     (df['persistent_workers'] == persistent)]
            if not data.empty:
                label = f'batch={batch_size}, persistent={persistent}'
                plt.plot(data['num_workers'], data['total_init_time'] * 1000,  # Convert to ms
                        marker='o', label=label)
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Initialization Time (ms)')
    plt.title('DataLoader Initialization Overhead')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'initialization_overhead.png')
    plt.close()
    
    # Detailed initialization breakdown
    plt.figure(figsize=(12, 8))
    width = 0.35
    
    for persistent in df['persistent_workers'].unique():
        data = df[df['persistent_workers'] == persistent]
        if not data.empty:
            offset = width/2 if persistent else -width/2
            x = np.arange(len(data['num_workers'].unique()))
            
            plt.bar(x + offset, data['init_overhead'] * 1000,
                   width, label=f'DataLoader Init (persistent={persistent})')
            plt.bar(x + offset, data['startup_overhead'] * 1000,
                   width, bottom=data['init_overhead'] * 1000,
                   label=f'Worker Startup (persistent={persistent})')
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (ms)')
    plt.title('Initialization Time Breakdown')
    plt.xticks(x, data['num_workers'].unique())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'initialization_breakdown.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Configure and run benchmarks
    batch_sizes = [64]
    num_workers_list = [0, 1, 4]
    persistent_workers_list = [False, True]
    num_batches = 100
    
    print("Starting DataLoader benchmarks...")
    results_df = run_benchmarks(
        batch_sizes=batch_sizes,
        num_workers_list=num_workers_list,
        persistent_workers_list=persistent_workers_list,
        num_batches=num_batches
    )
    
    # Save results
    output_dir = Path('benchmark_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_df.to_csv(output_dir / 'dataloader_benchmark_results.csv', index=False)
    
    # Generate and save plots
    plot_results(results_df, output_dir)
    
    # Print summary of best configurations
    print("\nTop 5 configurations by samples/sec:")
    print(results_df.nlargest(5, 'samples_per_sec')[
        ['batch_size', 'num_workers', 'persistent_workers', 'samples_per_sec']
    ].to_string())
    
    # Add initialization metrics to the summary
    print("\nTop 5 fastest initializing configurations:")
    print(results_df.nsmallest(5, 'total_init_time')[
        ['batch_size', 'num_workers', 'persistent_workers', 'total_init_time']
    ].to_string())
    
    print("\nDetailed initialization breakdown for best configuration:")
    best_config = results_df.nlargest(1, 'samples_per_sec').iloc[0]
    print(f"\nBest configuration:")
    print(f"Batch size: {best_config['batch_size']}")
    print(f"Num workers: {best_config['num_workers']}")
    print(f"Persistent workers: {best_config['persistent_workers']}")
    print(f"DataLoader init time: {best_config['init_overhead']*1000:.2f} ms")
    print(f"Worker startup time: {best_config['startup_overhead']*1000:.2f} ms")
    print(f"Total init time: {best_config['total_init_time']*1000:.2f} ms")
    print(f"Throughput: {best_config['samples_per_sec']:.2f} samples/sec")

if __name__ == '__main__':
    main() 