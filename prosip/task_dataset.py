#%%
from functools import partial
import logging
import concurrent.futures
import pickle
from pips.data import DATASET_LOADERS, ArcTask, ArrayTransform, ColorPermutation, DatasetType, Example
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from typing import NamedTuple, List, Dict
import hashlib
import random  # Add import for random sampling

logger = logging.getLogger(__name__)

#%%


class Tokenizer:
    def __init__(self, token2idx=None, idx2token=None, frozen=True) -> None:
        self.token2idx = token2idx if token2idx is not None else {}
        self.idx2token = idx2token if idx2token is not None else {}
        self.frozen = frozen
    
    def add_token(self, token):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def encode(self, sequence: str) -> List[int]:
        sequence = sequence.split(' ')
        return [self.token2idx[token] for token in sequence]

    def decode(self, sequence, remove_padding=True):
        tokens = [self.idx2token[idx] for idx in sequence]
        return ' '.join(tokens)
    
    def to_dict(self) -> Dict:
        return {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'frozen': self.frozen
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.token2idx = data['token2idx']
        obj.idx2token = data['idx2token']
        obj.frozen = data['frozen']
        return obj
    
    def __eq__(self, value: object) -> bool:
        assert isinstance(value, Tokenizer), 'value must be an instance of Tokenizer'
        return self.token2idx == value.token2idx and self.idx2token == value.idx2token

    def __len__(self):
        return len(self.token2idx)


class ProgramTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(frozen=False)

    def build(self, tokens: List[str]):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        for token in tokens:
            for t in token.strip().split(' '):
                if len(t) == 1:
                    print(f'Adding token: {token}')
                self.add_token(t)
        self.frozen = True


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


def load_task_loaders(loaders, cache_dir=Path(__file__).resolve().parent.parent / '.cache', verbose=False):
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
                 cache_dir=Path(__file__).resolve().parent.parent / '.cache',
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
        if not hasattr(self, '_num_original_tasks'):
            self._init_len_without_data()
            
        # Generate deterministic indices if needed
        if self.max_tasks is not None and self.max_tasks < self.num_of_total_tasks():
            self._generate_indices()

    def _init_len_without_data(self):
        """
        Initialize the number of original tasks without fully loading the data.
        This function quickly loads the grid data (using memory mapping)
        to compute the length and stores it as _num_original_tasks.
        """
        loaders = DATASET_LOADERS[self.dataset_type]
        temp_data = load_task_loaders(loaders, self.cache_dir, verbose=True)
        self._num_original_tasks = len(temp_data)
        del temp_data  # Clean up the temporary mapping


    @property
    def num_of_original_tasks(self):
        return self._num_original_tasks

    @property
    def num_of_total_tasks(self):
        return self._num_original_tasks * self.data_multiplier

    def _generate_indices(self):
        """
        Generate a deterministic random permutation of indices.
        This ensures we select the same subset of tasks across runs.
        """
        # Use a fixed seed for deterministic sampling
        rng = np.random.RandomState(self.seed)
        self.indices = rng.permutation(self.num_of_total_tasks)[:self.max_tasks]

    def _initialize_data(self):
        """Initialize the data for this process"""
        if self.data is None:
            loaders = DATASET_LOADERS[self.dataset_type]
            self.data = load_task_loaders(loaders, self.cache_dir)
            self._num_original_tasks = len(self.data)
            
            # Generate indices if needed and not already done
            if self.max_tasks is not None and self.max_tasks < self.num_of_total_tasks():
                if self.indices is None:
                    self._generate_indices()

    def __len__(self):
        actual_length = self.num_of_total_tasks

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
        if self.max_tasks is not None and self.max_tasks < self.num_of_total_tasks:
            idx = self.indices[idx]
        
        # Calculate the base index and augmentation round
        original_length = self.num_of_original_tasks
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

class ExampleDataset(Dataset):
    def __init__(self, dataset_type: DatasetType = DatasetType.TRAIN, 
                 cache_dir=Path(__file__).resolve().parent.parent / '.cache',
                 max_examples: int = None,
                 seed: int = 42,
                 data_multiplier: int = 1,
                 is_test: bool = False):
        """
        Dataset for ARC examples (individual input-output pairs) from tasks.
        
        Args:
            dataset_type: Type of dataset to load
            cache_dir: Directory to store cached data
            max_examples: Maximum number of examples to use (sample randomly if less than total)
            seed: Random seed for reproducibility
            data_multiplier: Factor to multiply dataset size by using deterministic augmentations
            is_test: Whether to use test examples (True) or train examples (False)
        """
        self.is_test = is_test
        self.seed = seed
        self.data_multiplier = data_multiplier
        
        # Create underlying TaskDataset
        self.task_dataset = TaskDataset(
            dataset_type=dataset_type,
            cache_dir=cache_dir,
            max_tasks=None,  # We'll handle max_examples ourselves
            seed=seed,
            data_multiplier=data_multiplier  # Use same multiplier
        )
        
        # Set examples per task based on is_test flag
        self.examples_per_task = 1 if is_test else 3
                
        # Calculate max examples and handle indices
        self.max_examples = max_examples
        self.indices = None
        
        # Generate deterministic indices if needed
        if self.max_examples is not None and self.max_examples < len(self):
            self._generate_indices()
    
    @property
    def number_of_original_examples(self):
        """Initialize the base length without fully loading data"""
        # Get task dataset's base length (before multiplier)
        base_task_length = self.task_dataset._num_original_tasks
        return base_task_length * self.examples_per_task
    
    @property
    def number_of_total_examples(self):
        """Get the actual length without max_examples constraint"""
        return self.number_of_original_examples * self.data_multiplier
    
    def _generate_indices(self):
        """Generate deterministic random indices for sampling max_examples"""
        rng = np.random.RandomState(self.seed)
        self.indices = rng.permutation(self.number_of_total_examples)[:self.max_examples]
    
    def __len__(self):
        """Return the number of examples in the dataset"""
        actual_length = self.number_of_total_examples
        
        # Apply max_examples limit if specified
        if self.max_examples is not None:
            actual_length = min(actual_length, self.max_examples)
            
        return actual_length
    
    def _initialize_data(self):
        """Initialize the data for this process"""
        self.task_dataset._initialize_data()
        # Generate indices if needed and not already done
        if self.max_examples is not None and self.max_examples < self.number_of_total_examples:
            if self.indices is None:
                self._generate_indices()
    
    def __getitem__(self, idx) -> Example:
        """Get a specific example from a task"""
        # Initialize data if not already done
        if self.task_dataset.data is None:
            self._initialize_data()
        
        # If we're using a subset of indices, map the index
        if self.max_examples is not None and self.max_examples < self.number_of_total_examples:
            idx = self.indices[idx]
        
        # Calculate original length and augmentation round (same logic as TaskDataset)
        original_length = self.number_of_original_examples
        augmentation_round = idx // original_length
        base_idx = idx % original_length
                
        # Calculate which task and which example within task
        task_idx = base_idx // self.examples_per_task
        example_idx = base_idx % self.examples_per_task
        
        # Get the task from the TaskDataset, which handles permutation internally
        task = self.task_dataset[task_idx + augmentation_round * self.task_dataset.num_of_original_tasks]
        
        # Return the appropriate example based on is_test flag
        if self.is_test:
            return task.test[example_idx]
        else:
            return task.train[example_idx]
    
    def unload(self):
        """Reset the internal state of the dataset"""
        self.task_dataset.unload()
        self.indices = None


    def get_program_tokenizer(self):
        filename = f'program_tokenizer_{self.task_dataset.dataset_type.name}_{self.data_multiplier}_{self.is_test}.pkl'
        cache_file = self.task_dataset.cache_dir / filename

        if cache_file.exists():
            print(f"Loading program tokenizer from {cache_file}")
            return pickle.load(cache_file.open('rb'))

        tokenizer = ProgramTokenizer()
        program_ids = []
        for idx in tqdm(range(len(self)), desc="Building program tokenizer"):
            program_ids.append(self[idx].uid)
        tokenizer.build(program_ids)

        with open(cache_file, 'wb') as f:
            print(f"Saving program tokenizer to {cache_file}")
            pickle.dump(tokenizer, f)
        return tokenizer

# Update the worker_init_fn to be simpler
def worker_init_fn(worker_id):
    """Initialize worker for DataLoader"""
    # print(f"Initializing worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._initialize_data()


# Define the NamedTuple for the collate function output
class EXAMPLE(NamedTuple):
    input_grids: torch.Tensor
    output_grids: torch.Tensor
    program_ids: torch.Tensor
    attributes: List[Dict[str, any]]



def collate_fn_project(batch, program_tokenizer, pad_value=-1, device=torch.device('cpu'), max_height=32, max_width=32, flatten=True):    
    flattened_input_grids = []
    flattened_output_grids = []
    program_ids = []
    attributes = []

    for example in batch:
        # Flatten each grid with optional EOS markers and padding

        input_grids, output_grids = example.input, example.output

        projected_input_grids = input_grids.project(new_height=max_height, new_width=max_width, pad_value=pad_value)
        projected_output_grids = output_grids.project(new_height=max_height, new_width=max_width, pad_value=pad_value)

        flattened_input_grids.append(projected_input_grids.flatten() if flatten else projected_input_grids)
        flattened_output_grids.append(projected_output_grids.flatten() if flatten else projected_output_grids)

        program_ids.append(program_tokenizer.encode(example.uid)[0])

        # Collect attributes and convert numpy types to native Python types
        attributes.append({
            'idx': int(example.idx),
            'program_id': str(example.program_id),
            'task_id': str(example.task_id),
            'dataset': str(example.dataset),
            'color_perm': str(example.color_perm),
            'transform': str(example.transform),
            'is_test': bool(example.is_test),
        })

    # Convert the lists of numpy arrays to tensors
    flattened_input_grids = np.array(flattened_input_grids)
    flattened_output_grids = np.array(flattened_output_grids)
    program_ids = np.array(program_ids)


    # Convert to torch tensors and move to device
    flattened_input_grids = torch.tensor(flattened_input_grids, dtype=torch.long, requires_grad=False).to(device, non_blocking=True)
    flattened_output_grids = torch.tensor(flattened_output_grids, dtype=torch.long, requires_grad=False).to(device, non_blocking=True)
    program_ids = torch.tensor(program_ids, dtype=torch.long, requires_grad=False).to(device, non_blocking=True)
    return EXAMPLE(input_grids=flattened_input_grids, 
                   output_grids=flattened_output_grids, 
                   program_ids=program_ids, 
                   attributes=attributes)




if __name__ == "__main__":

    data_multiplier = 2
    dataset_type = DatasetType.ALL
    is_test = False
    ds = TaskDataset(dataset_type=dataset_type, data_multiplier=data_multiplier)
    dss = TaskDataset(dataset_type=DatasetType.ALL_SMALL, data_multiplier=data_multiplier)
    print("Length of dataset:", len(ds))
    print("Number of original tasks:", ds.num_of_original_tasks)
    print("Number of total tasks:", ds.num_of_total_tasks)

    assert ds.num_of_total_tasks == data_multiplier * ds.num_of_original_tasks

    # Test TaskDataset
    print(ds[0])
    print(ds[0+ds.num_of_original_tasks])
    print(ds[0+ds.num_of_original_tasks].train[0])
    print(ds[0+ds.num_of_original_tasks].train[0].input)

    print(ds[1])
    print(ds[1+ds.num_of_original_tasks])
    print(ds[1+ds.num_of_original_tasks].train[0])
    print(ds[1+ds.num_of_original_tasks].train[0].input)
    
    
    # Create train example dataset
    train_ex_ds = ExampleDataset(dataset_type=dataset_type, data_multiplier=data_multiplier, is_test=is_test)
    print("Length of train example dataset:", len(train_ex_ds))
    print("Train examples per task:", train_ex_ds.examples_per_task)
    print("Base length:", train_ex_ds.number_of_original_examples)
    print("Total length:", train_ex_ds.number_of_total_examples)

    assert train_ex_ds.number_of_original_examples == train_ex_ds.examples_per_task * ds.num_of_original_tasks
    assert train_ex_ds.number_of_total_examples == train_ex_ds.examples_per_task * ds.num_of_total_tasks
    assert train_ex_ds.number_of_total_examples == data_multiplier * train_ex_ds.number_of_original_examples

    tokenizer = train_ex_ds.get_program_tokenizer()

    print("Number of tokens in tokenizer:", len(tokenizer))
    print("Number of examples in train example dataset:", train_ex_ds.number_of_total_examples // 3)
    if len(tokenizer) < train_ex_ds.number_of_total_examples // train_ex_ds.examples_per_task:
        print(f"Warning: Number of unique tokens ({len(tokenizer)}) is less than expected unique tasks "
              f"({train_ex_ds.number_of_total_examples // train_ex_ds.examples_per_task}). "
              f"This is likely due to permutation collisions.")

    # # Create test example dataset
    # test_ex_ds = ExampleDataset(dataset_type=DatasetType.ALL, data_multiplier=2, is_test=True)
    # print("Length of test example dataset:", len(test_ex_ds))
    # print("Test examples per task:", test_ex_ds.examples_per_task)
    # print("Base length:", test_ex_ds._number_of_original_examples)
    
    # Test accessing examples

    train_idx = 6
    print("\nExample from train dataset (original, unpermuted):")
    train_example = train_ex_ds[train_idx]
    print(train_example)

    print("UID:", train_example.uid)
    print("Tokenized UID:", tokenizer.encode(train_example.uid))

    print(tokenizer.idx2token[1])
    print(tokenizer.idx2token[0])

    print("\nExample from train dataset (permuted):")
    train_example = train_ex_ds[train_ex_ds.number_of_original_examples + train_idx]
    print(train_example)
    
    # Test with DataLoader
    print("\nTesting with DataLoader:")

    collate_fn = partial(collate_fn_project, 
                         program_tokenizer=tokenizer,
                         pad_value=-1,
                         max_height=32,
                         max_width=32,
                         flatten=False,
                         device=torch.device('cpu'))

    train_loader = DataLoader(
        train_ex_ds, 
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        shuffle=False
    )

    input_grids, output_grids, program_ids, attributes = next(iter(train_loader))
    print(input_grids.shape)
    print(output_grids.shape)
    print(program_ids.shape)
    print(attributes)

    print(program_ids)
    for pid in program_ids:
        print(tokenizer.decode([pid.item()]))