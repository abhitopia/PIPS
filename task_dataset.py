#%%
import logging
import concurrent.futures
from pips.data import DATASET_LOADERS, ArcTask, ArrayTransform, ColorPermutation, DatasetType, Example, Grid, ARCAGI1_TRAIN
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from typing import NamedTuple, List, Dict
import hashlib
import random  # Add import for random sampling

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

#%%



def process_task_loader(loader, output_file, num_train=3, num_test=1):
    if output_file.exists():
        logger.info(f"Cache exists for {loader.name} at {output_file}. Skipping.")
        return

    tasks = loader.tasks

    logger.info(f"Loaded {len(tasks)} grids for {loader.name}")

    # Define the numpy dtype for task
    task_dtype = np.dtype([
        # Task metadata
        ('task_id', 'U50'),
        ('program_id', 'U50'),
        ('dataset', 'U50'),
        ('color_perm', 'U50'),
        ('transform', 'U50'),
        
        # Train pairs (input and output grids)
        *[(f'train_input_{i}', (np.int8, (30, 30))) for i in range(1, num_train+1)],
        *[(f'train_output_{i}', (np.int8, (30, 30))) for i in range(1, num_train+1)],
        
        # Test pairs (input and output grids)
        *[(f'test_input_{i}', (np.int8, (30, 30))) for i in range(1, num_test+1)],
        *[(f'test_output_{i}', (np.int8, (30, 30))) for i in range(1, num_test+1)],
        
        # Original shapes for each grid
        *[(f'train_input_{i}_shape_h', np.int32) for i in range(1, num_train+1)],
        *[(f'train_input_{i}_shape_w', np.int32) for i in range(1, num_train+1)],
        *[(f'train_output_{i}_shape_h', np.int32) for i in range(1, num_train+1)],
        *[(f'train_output_{i}_shape_w', np.int32) for i in range(1, num_train+1)],
        *[(f'test_input_{i}_shape_h', np.int32) for i in range(1, num_test+1)],
        *[(f'test_input_{i}_shape_w', np.int32) for i in range(1, num_test+1)],
        *[(f'test_output_{i}_shape_h', np.int32) for i in range(1, num_test+1)],
        *[(f'test_output_{i}_shape_w', np.int32) for i in range(1, num_test+1)]
    ])
    
    # Filter tasks with required number of examples
    valid_tasks = []
    for task in tasks:
        # Find valid training examples (where both input and output are at most 30x30)
        valid_train_examples = []
        for example in task.train:
            if (example.input.array.shape[0] <= 30 and 
                example.input.array.shape[1] <= 30 and
                example.output.array.shape[0] <= 30 and
                example.output.array.shape[1] <= 30):
                valid_train_examples.append(example)
        
        # Find valid test examples (where both input and output are at most 30x30)
        valid_test_examples = []
        for example in task.test:
            if (example.input.array.shape[0] <= 30 and 
                example.input.array.shape[1] <= 30 and
                example.output.array.shape[0] <= 30 and
                example.output.array.shape[1] <= 30):
                valid_test_examples.append(example)
        
        # Only include tasks with enough valid examples
        if len(valid_train_examples) >= num_train and len(valid_test_examples) >= num_test:
            # Randomly select examples at this stage
            task.selected_train = random.sample(valid_train_examples, num_train)
            task.selected_test = random.sample(valid_test_examples, num_test)
            valid_tasks.append(task)
    
    logger.info(f"Found {len(valid_tasks)} valid tasks with at least {num_train} train and {num_test} test examples (max 30x30)")
    
    if not valid_tasks:
        logger.warning(f"No valid tasks found for {loader.name}. Skipping.")
        return
    
    task_data = np.zeros(len(valid_tasks), dtype=task_dtype)
    
    for i, task in tqdm(enumerate(valid_tasks), total=len(valid_tasks), desc=f"Processing tasks for {loader.name}"):
        # Create dictionary with keys matching the dtype field names
        task_dict = {}
        
        # Add metadata
        task_dict['task_id'] = task.id
        task_dict['program_id'] = task.prog_id
        task_dict['dataset'] = task.dataset
        task_dict['color_perm'] = task._color_perm.name
        task_dict['transform'] = task._transform.name
        
        # Process selected train examples (already randomly selected)
        for j in range(num_train):
            example = task.selected_train[j]
            
            # Input grid
            in_grid = example.input.array
            in_shape = in_grid.shape
            
            # Create padded grid filled with -1
            padded_in = np.full((30, 30), -1, dtype=np.int8)
            padded_in[:in_shape[0], :in_shape[1]] = in_grid
            
            task_dict[f'train_input_{j+1}'] = padded_in
            task_dict[f'train_input_{j+1}_shape_h'] = in_shape[0]
            task_dict[f'train_input_{j+1}_shape_w'] = in_shape[1]
            
            # Output grid
            out_grid = example.output.array
            out_shape = out_grid.shape
            
            # Create padded grid filled with -1
            padded_out = np.full((30, 30), -1, dtype=np.int8)
            padded_out[:out_shape[0], :out_shape[1]] = out_grid
            
            task_dict[f'train_output_{j+1}'] = padded_out
            task_dict[f'train_output_{j+1}_shape_h'] = out_shape[0]
            task_dict[f'train_output_{j+1}_shape_w'] = out_shape[1]
        
        # Process selected test examples (already randomly selected)
        for j in range(num_test):
            example = task.selected_test[j]
            
            # Input grid
            in_grid = example.input.array
            in_shape = in_grid.shape
            
            # Create padded grid filled with -1
            padded_in = np.full((30, 30), -1, dtype=np.int8)
            padded_in[:in_shape[0], :in_shape[1]] = in_grid
            
            task_dict[f'test_input_{j+1}'] = padded_in
            task_dict[f'test_input_{j+1}_shape_h'] = in_shape[0]
            task_dict[f'test_input_{j+1}_shape_w'] = in_shape[1]
            
            # Output grid
            out_grid = example.output.array
            out_shape = out_grid.shape
            
            # Create padded grid filled with -1
            padded_out = np.full((30, 30), -1, dtype=np.int8)
            padded_out[:out_shape[0], :out_shape[1]] = out_grid
            
            task_dict[f'test_output_{j+1}'] = padded_out
            task_dict[f'test_output_{j+1}_shape_h'] = out_shape[0]
            task_dict[f'test_output_{j+1}_shape_w'] = out_shape[1]
        
        # Assign the dictionary directly to the structured array
        for field_name in task_dtype.names:
            task_data[i][field_name] = task_dict[field_name]
    
    np.save(output_file, task_data)
    logger.info(f"Saved {len(valid_tasks)} tasks to {output_file} for {loader.name}")


def load_task_loaders(loaders, cache_dir=Path('.cache'), verbose=False):
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sort loaders by name
    sorted_loaders = sorted(loaders, key=lambda loader: loader.name)

    # Create a hash from the sorted loader names
    loader_names = ''.join(loader.name for loader in sorted_loaders)
    unified_file_hash = hashlib.md5(loader_names.encode()).hexdigest()
    unified_file = cache_dir / f'task_data_{unified_file_hash}.npy'

    if not unified_file.exists():
        output_files = {}
        for loader in sorted_loaders:
            output_file = cache_dir / f"{loader.name}_task_data.npy"
            output_files[loader.name] = output_file

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_task_loader, loader, output_files[loader.name]): loader for loader in sorted_loaders}

            for future in concurrent.futures.as_completed(futures):
                loader = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {loader.name}: {e}")

        all_data = []

        if verbose:
            logger.info(f"Saving Task data to cache...")
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



class TaskDataset(Dataset):
    def __init__(self, dataset_type: DatasetType = DatasetType.TRAIN, 
                 cache_dir=Path(__file__).resolve().parent / '.cache',
                 max_tasks: int = None,
                 seed: int = 42,
                 data_multiplier: int = 1):
        """
        Args:
            dataset_type: Type of dataset to load
            cache_dir: Directory to store cached data
            max_tasks: Maximum number of tasks to use (sample randomly if less than total)
            seed: Random seed for reproducibility
            data_multiplier: Factor to multiply dataset size by using deterministic augmentations
        """
        assert data_multiplier >= 1, "data_multiplier must be at least 1"
        
        self.dataset_type = dataset_type
        self.cache_dir = cache_dir
        self.data = None
        self.max_tasks = max_tasks
        self.seed = seed
        self.data_multiplier = data_multiplier
        self.indices = None  # Will store our sampled indices
        
        # Initialize shared length if not present.
        if not hasattr(self, '_shared_length'):
            self._init_len_without_data()
            
        # Generate deterministic indices if needed
        if self.max_tasks is not None and self.max_tasks < self._actual_length():
            self._generate_indices()

    def _init_len_without_data(self):
        """
        Initialize the shared length without fully loading the data.
        This function quickly loads the grid data (using memory mapping)
        to compute the length and stores it as _shared_length.
        """
        loaders = DATASET_LOADERS[self.dataset_type]
        temp_data = load_task_loaders(loaders, self.cache_dir, verbose=True)
        self._shared_length = len(temp_data)
        del temp_data  # Clean up the temporary mapping


    def _actual_length(self):
        return self._shared_length * self.data_multiplier

    def _generate_indices(self):
        """
        Generate a deterministic random permutation of indices.
        This ensures we select the same subset of tasks across runs.
        """
        # Use a fixed seed for deterministic sampling
        rng = np.random.RandomState(self.seed)
        self.indices = rng.permutation(self._actual_length())[:self.max_tasks]

    def _initialize_data(self):
        """Initialize the data for this process"""
        if self.data is None:
            loaders = DATASET_LOADERS[self.dataset_type]
            self.data = load_task_loaders(loaders, self.cache_dir)
            self._shared_length = len(self.data)
            
            # Generate indices if needed and not already done
            if self.max_tasks is not None and self.max_tasks < len(self):
                if self.indices is None:
                    self._generate_indices()
            # print(f"Initialized data with {len(self.data)} grids for {self.dataset_type.name}")

    def __len__(self):
        actual_length = self._actual_length()

        # Apply max_tasks limit if specified
        if self.max_tasks is not None:
            actual_length = min(actual_length, self.max_tasks)
            
        # Multiply by data_multiplier to account for augmentations
        return actual_length

    def __getitem__(self, idx):
        # Initialize data if not already done
        if self.data is None:
            self._initialize_data()


        # If we're using a subset of indices, map the base index to our selected indices
        if self.max_tasks is not None and self.max_tasks < self._actual_length():
            idx = self.indices[idx]
        
        # Calculate the base index and augmentation round
        original_length = len(self.data)
        augmentation_round = idx // original_length
        base_idx = idx % original_length
        
        # Get the original task
        task = self.data[base_idx]

        # reconstruct the task from the data
        train_examples = []
        test_examples = []
        
        # Process training examples
        num_train = 3  # Based on the fixed size used in process_task_loader
        for i in range(1, num_train + 1):
            # Get input and output grids with their original shapes
            input_grid = task[f'train_input_{i}']
            output_grid = task[f'train_output_{i}']
            
            # Get original shapes
            input_h = task[f'train_input_{i}_shape_h'] 
            input_w = task[f'train_input_{i}_shape_w']
            output_h = task[f'train_output_{i}_shape_h']
            output_w = task[f'train_output_{i}_shape_w']
            
            # Extract actual grids using original shapes
            input_array = input_grid[:input_h, :input_w]
            output_array = output_grid[:output_h, :output_w]
            
            # Create Example object
            example = Example(
                idx=i-1,
                input=input_array,
                output=output_array,
                program_id=task['program_id'],
                task_id=task['task_id'],
                dataset=task['dataset'],
                color_perm=ColorPermutation.from_name(task['color_perm']),  # Original color permutation
                transform=ArrayTransform[task['transform']],    # Original transformation
                is_test=False
            )
            train_examples.append(example)
        
        # Process test examples
        num_test = 1  # Based on the fixed size used in process_task_loader
        for i in range(1, num_test + 1):
            # Get input and output grids with their original shapes
            input_grid = task[f'test_input_{i}']
            output_grid = task[f'test_output_{i}']
            
            # Get original shapes
            input_h = task[f'test_input_{i}_shape_h']
            input_w = task[f'test_input_{i}_shape_w']
            output_h = task[f'test_output_{i}_shape_h']
            output_w = task[f'test_output_{i}_shape_w']
            
            # Extract actual grids using original shapes
            input_array = input_grid[:input_h, :input_w]
            output_array = output_grid[:output_h, :output_w]
            
            # Create Example object
            example = Example(
                idx=i-1,
                input=input_array,
                output=output_array,
                program_id=task['program_id'],
                task_id=task['task_id'],
                dataset=task['dataset'],
                color_perm=ColorPermutation.from_name(task['color_perm']),  # Original color permutation
                transform=ArrayTransform[task['transform']],    # Original transformation
                is_test=True
            )
            test_examples.append(example)
        
        # Create ArcTask object
        arc_task = ArcTask(
            id=task['task_id'],
            prog_id=task['program_id'],
            train=train_examples,
            test=test_examples,
            dataset=task['dataset'],
            color_perm=ColorPermutation.from_name(task['color_perm']),
            transform=ArrayTransform[task['transform']]
        )
        
        # Apply permutation if this is an augmented version (not the original data)
        if augmentation_round > 0:
            # Create a unique permutation for this augmentation
            arc_task = arc_task.permute(seed=idx)
        
        return arc_task

    def unload(self):
        """
        Reset the internal state of the dataset to simulate a never-initialized condition.
        After calling this, the dataset will behave as if neither __len__ nor __getitem__
        has ever been called.
        """
        self.data = None
        self.indices = None

# Update the worker_init_fn to be simpler
def worker_init_fn(worker_id):
    """Initialize worker for DataLoader"""
    # print(f"Initializing worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._initialize_data()  # Initialize data for this worker


if __name__ == "__main__":
    ds = TaskDataset(dataset_type=DatasetType.ALL, data_multiplier=2)

    print("Length of dataset:", len(ds))
    print("Shared length:", ds._shared_length)

    print(ds[0])
    print(ds[1])
    print(ds[0+ds._shared_length])
    print(ds[0+ds._shared_length].train[0])
    print(ds[0+ds._shared_length].train[0].input)

    
    print(ds[1+ds._shared_length])
    print(ds[1+ds._shared_length].train[0])
    print(ds[1+ds._shared_length].train[0].input)