import logging
import concurrent.futures
from pips.data import DATASET_LOADERS, DatasetType, Grid
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from typing import NamedTuple, List, Dict
import hashlib

logger = logging.getLogger(__name__)



# Define a structured data type for the combined grid and attributes
combined_dtype = np.dtype([
    ('grid', (np.int8, (30, 30))),  # Assuming grid size is 30x30
    ('idx', np.int32),
    ('program_id', 'U50'),  # Assuming max length of 50
    ('task_id', 'U50'),
    ('dataset', 'U50'),
    ('color_perm', 'U10'),
    ('transform', 'U10'),
    ('is_test', np.bool_),
    ('is_input', np.bool_),
    ('original_shape', (np.int32, 2))  # Assuming shape is a tuple of two integers
])

def process_grid_loader(loader, output_file):
    """Function to load tasks for a specific loader and save to a single NPY file."""
    if output_file.exists():
        logger.info(f"Cache exists for {loader.name}. Skipping.")
        return

    grids = loader.train_grids + loader.test_grids

    # Log the number of grids loaded
    logger.info(f"Loaded {len(grids)} grids for {loader.name}")

    # Check if grids are empty
    if not grids:
        logger.warning(f"No grids found for {loader.name}. Check the data source.")
        return

    combined_data = np.zeros(len(grids), dtype=combined_dtype)

    valid_grid_count = 0
    for i, grid in tqdm(enumerate(grids), total=len(grids), desc=f"Processing grids for {loader.name}"):
        if grid.array.shape[0] > 30 or grid.array.shape[1] > 30:
            logger.debug(f"Skipping grid {grid.idx} with shape {grid.array.shape} for {loader.name}")
            continue

        # Initialize a 30x30 grid with -1
        grid_data = np.full((30, 30), -1, dtype=np.int8)
        
        # Copy the smaller grid into the larger grid
        grid_data[:grid.array.shape[0], :grid.array.shape[1]] = grid.array

        combined_data[valid_grid_count] = (
            grid_data,
            grid.idx,
            grid.program_id,
            grid.task_id,
            grid.dataset,
            grid.color_perm,
            grid.transform,
            grid.is_test,
            grid.is_input,
            grid.array.shape
        )
        valid_grid_count += 1

    if valid_grid_count == 0:
        logger.warning(f"No valid grids processed for {loader.name}. Check grid dimensions and filtering criteria.")
        return

    # Resize combined_data to only include valid grids
    combined_data = combined_data[:valid_grid_count]

    np.save(output_file, combined_data)
    logger.info(f"Saved {valid_grid_count} valid grids to {output_file} for {loader.name}")

def load_grid_loaders(loaders, cache_dir=Path(__file__).resolve().parent.parent / '.cache', verbose=False):
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sort loaders by name
    sorted_loaders = sorted(loaders, key=lambda loader: loader.name)

    # Create a hash from the sorted loader names
    loader_names = ''.join(loader.name for loader in sorted_loaders)
    unified_file_hash = hashlib.md5(loader_names.encode()).hexdigest()
    unified_file = cache_dir / f'grid_data_{unified_file_hash}.npy'

    if not unified_file.exists():
        output_files = {}
        for loader in sorted_loaders:
            output_file = cache_dir / f"{loader.name}_grid_data.npy"
            output_files[loader.name] = output_file

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_grid_loader, loader, output_files[loader.name]): loader for loader in sorted_loaders}

            for future in concurrent.futures.as_completed(futures):
                loader = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {loader.name}: {e}")

        all_data = []

        if verbose:
            logger.info(f"Saving grid data to cache...")
        for loader_name, output_file in output_files.items():
            assert output_file.exists(), f"Output file for {loader_name} does not exist"
            data = np.load(output_file, mmap_mode='r')
            all_data.append(data)

        all_data = np.concatenate(all_data, axis=0)

        np.save(unified_file, all_data)
    else:
        if verbose:
            logger.info(f"Loading grid data from cache...")
        all_data = np.load(unified_file, mmap_mode='r')

    return all_data

# Define the NamedTuple for the collate function output
class GRID_INPUT(NamedTuple):
    grids: torch.Tensor  # (B, S) where S is the flattened size of the projected grid
    attributes: List[Dict[str, any]]
    # positions: torch.Tensor = None # (B, S, 2) where S is the flattened size of the projected grid


class GridDataset(Dataset):
    def __init__(self, dataset_type: DatasetType = DatasetType.TRAIN, 
                 cache_dir=Path(__file__).resolve().parent.parent / '.cache', 
                 max_samples: int = None,
                 min_samples: int = None):
        self.dataset_type = dataset_type
        self.cache_dir = cache_dir
        self.max_samples = max_samples  # New parameter to limit available samples
        self.min_samples = min_samples  # New parameter to set minimum dataset length
        self.data = None
        
        # Initialize shared length if not present.
        if not hasattr(self, '_shared_length'):
            self._init_len_without_data()

    def _init_len_without_data(self):
        """
        Initialize the shared length without fully loading the data.
        This function quickly loads the grid data (using memory mapping)
        to compute the length and stores it as _shared_length.
        """
        loaders = DATASET_LOADERS[self.dataset_type]
        temp_data = load_grid_loaders(loaders, self.cache_dir, verbose=True)
        self._shared_length = len(temp_data)
        del temp_data  # Clean up the temporary mapping

    def _initialize_data(self):
        """Initialize the data for this process"""
        if self.data is None:
            loaders = DATASET_LOADERS[self.dataset_type]
            self.data = load_grid_loaders(loaders, self.cache_dir)
            self._shared_length = len(self.data)
            # print(f"Initialized data with {len(self.data)} grids for {self.dataset_type.name}")

    def __len__(self):
        actual_length = self._shared_length
        
        # Apply max_samples limit if specified
        if self.max_samples is not None:
            actual_length = min(actual_length, self.max_samples)
            
        # Apply min_samples requirement if specified
        if self.min_samples is not None:
            actual_length = max(actual_length, self.min_samples)
            
        return actual_length

    def __getitem__(self, idx):
        # Initialize data if not already done
        if self.data is None:
            self._initialize_data()
        
        # If idx is beyond the actual data length, wrap around to reuse samples
        if self.min_samples is not None and idx >= self._shared_length:
            idx = idx % self._shared_length
            
        sample = self.data[idx]
        
        # Extract the original shape
        original_shape = sample['original_shape']
        
        # Reshape the grid to its original dimensions
        grid_array = sample['grid'][:original_shape[0], :original_shape[1]]
        
        # Create a Grid object using the attributes from the sample
        grid = Grid(
            array=grid_array,
            idx=sample['idx'],
            program_id=sample['program_id'],
            task_id=sample['task_id'],
            dataset=sample['dataset'],
            color_perm=sample['color_perm'],
            transform=sample['transform'],
            is_test=sample['is_test'],
            is_input=sample['is_input']
        )
        
        return grid
    

    @staticmethod
    def collate_fn_project(batch, pad_value=-1, device=torch.device('cpu'), permute=False, max_height=32, max_width=32, flatten=True) -> GRID_INPUT:
        """Collate function to process a batch of Grids.

        Args:
            batch (list of Grid): The batch of Grid objects.
            pad_value (int): The value to use for padding. Default is -1.
            device (str or torch.device): The device to move the tensors to. Default is 'cpu'.
            permute (bool): Whether to permute the grid before projection. Default is False.
            max_height (int): Maximum height for the projected grid. Default is 32.
            max_width (int): Maximum width for the projected grid. Default is 32.
            eos_value (Optional[int]): Value to use for end-of-sequence markers. If None, no EOS markers are added.

        Returns:
            GRID_INPUT: A named tuple containing the flattened grids, position indices, and attributes.
        """
        flattened_grids = []
        attributes = []

        for grid in batch:
            # Optionally permute the grid using the grid.permute method
            if permute:
                grid = grid.permute()

            # Flatten each grid with optional EOS markers and padding
            projected_array = grid.project(new_height=max_height, new_width=max_width, pad_value=pad_value)
            flattened_grids.append(projected_array.flatten() if flatten else projected_array)

            # Collect attributes and convert numpy types to native Python types
            attributes.append({
                'idx': int(grid.idx),
                'program_id': str(grid.program_id),
                'task_id': str(grid.task_id),
                'dataset': str(grid.dataset),
                'color_perm': str(grid.color_perm),
                'transform': str(grid.transform),
                'is_test': bool(grid.is_test),
                'is_input': bool(grid.is_input)
            })

        # Convert the lists of numpy arrays to tensors
        flattened_grids = np.array(flattened_grids)

        # Convert to torch tensors and move to device
        flattened_grids = torch.tensor(flattened_grids, dtype=torch.long, requires_grad=False).to(device, non_blocking=True)

        return GRID_INPUT(grids=flattened_grids, attributes=attributes)


    @staticmethod
    def collate_fn_flatten(batch, pad_value=-1, device=torch.device('cpu'), permute=False, max_size=1024, eos_value=None) -> GRID_INPUT:
        """Collate function to process a batch of Grids.

        Args:
            batch (list of Grid): The batch of Grid objects.
            pad_value (int): The value to use for padding. Default is -1.
            device (str or torch.device): The device to move the tensors to. Default is 'cpu'.
            permute (bool): Whether to permute the grid before projection. Default is False.
            max_size (int): Maximum length for the flattened arrays. Default is 1024.
            eos_value (Optional[int]): Value to use for end-of-sequence markers. If None, no EOS markers are added.

        Returns:
            GRID_INPUT: A named tuple containing the flattened grids, position indices, and attributes.
        """
        flattened_grids = []
        flattened_positions = []
        attributes = []

        for grid in batch:
            # Optionally permute the grid using the grid.permute method
            if permute:
                grid = grid.permute()

            # Flatten each grid with optional EOS markers and padding
            flattened_array, positions = grid.flatten(max_size=max_size, pad_value=pad_value, eos_value=eos_value)
            flattened_grids.append(flattened_array)
            flattened_positions.append(positions)

            # Collect attributes and convert numpy types to native Python types
            attributes.append({
                'idx': int(grid.idx),
                'program_id': str(grid.program_id),
                'task_id': str(grid.task_id),
                'dataset': str(grid.dataset),
                'color_perm': str(grid.color_perm),
                'transform': str(grid.transform),
                'is_test': bool(grid.is_test),
                'is_input': bool(grid.is_input)
            })

        # Convert the lists of numpy arrays to tensors
        flattened_grids = np.array(flattened_grids)
        flattened_positions = np.array(flattened_positions)
        
        # Convert to torch tensors and move to device
        flattened_grids = torch.tensor(flattened_grids, dtype=torch.long, requires_grad=False).to(device, non_blocking=True)
        flattened_positions = torch.tensor(flattened_positions, dtype=torch.long, requires_grad=False).to(device, non_blocking=True)

        return GRID_INPUT(grids=flattened_grids, attributes=attributes, positions=flattened_positions)

    def unload(self):
        """
        Reset the internal state of the dataset to simulate a never-initialized condition.
        After calling this, the dataset will behave as if neither __len__ nor __getitem__
        has ever been called.
        """
        self.data = None

# Update the worker_init_fn to be simpler
def worker_init_fn(worker_id):
    """Initialize worker for DataLoader"""
    # print(f"Initializing worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._initialize_data()  # Initialize data for this worker

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format to just show the message
    )

    dataset = GridDataset(dataset_type=DatasetType.ARC_TRAIN)
    print(f"Total number of samples in dataset: {len(dataset)}")

    # Create a DataLoader with batch size 32, pad_value 15, and random shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: GridDataset.collate_fn_project(x, pad_value=15, device='cpu', permute=True),
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    # Iterate over the DataLoader
    for batch in dataloader:
        print(f"Batch grids shape: {batch.grids.shape}")  # Should be (32, 32, 32)
        # print(f"Batch attributes: {batch.attributes}")
        break  # Just process the first batch for demonstration

