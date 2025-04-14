#%%
import copy
from enum import Enum, auto
import logging
from pathlib import Path
import random
from typing import List, Optional
import numpy as np
from tqdm import tqdm
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)

try:
    import ujson as json
    logger.info("Using ujson for faster JSON parsing")
except ImportError:
    import json
    logger.warning("ujson not found, falling back to standard json module. Consider installing ujson for better performance: pip install ujson")


class ColorPermutation(Enum):
    CPID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Identity
    CP01 = [7, 4, 5, 2, 8, 3, 0, 9, 6, 1]
    CP02 = [0, 9, 4, 5, 6, 8, 1, 3, 2, 7]
    CP03 = [7, 4, 1, 9, 6, 0, 8, 2, 5, 3]
    CP04 = [9, 6, 5, 7, 4, 0, 3, 8, 1, 2]
    CP05 = [1, 8, 0, 3, 9, 5, 6, 2, 7, 4]
    CP06 = [5, 3, 1, 9, 7, 6, 0, 2, 8, 4]
    CP07 = [1, 4, 3, 8, 7, 9, 6, 2, 5, 0]
    CP08 = [6, 0, 2, 1, 3, 4, 7, 8, 5, 9]
    CP09 = [2, 0, 3, 8, 4, 6, 1, 9, 5, 7]
    RAND = None  # Placeholder for random permutation

    @property
    def transform(self):
        if self == ColorPermutation.RAND:
            # Generate a random permutation of numbers 0-9
            colors = list(range(10))
            random.shuffle(colors)
            color_mapping = {original: new for original, new in enumerate(colors)}
            return lambda x: np.vectorize(color_mapping.get)(x)
        else:
            colors = self.value
            color_mapping = {original: new for original, new in enumerate(colors)}
            return lambda x: np.vectorize(color_mapping.get)(x)


class ArrayTransform(Enum):
    IDENT = auto()
    RT090 = auto()
    RT180 = auto()
    RT270 = auto()
    FLPLR = auto()
    FLPUD = auto()
    FLPDG = auto()
    FLPAD = auto()

    @property
    def transform(self):
        return {
            'IDENT': lambda x: x,
            'RT090': lambda x: np.rot90(x),
            'RT180': lambda x: np.rot90(x, k=2),
            'RT270' : lambda x: np.rot90(x, k=3),
            'FLPLR': lambda x: np.fliplr(x),
            'FLPUD': lambda x: np.flipud(x),
            'FLPDG': lambda x: np.flipud(np.rot90(x)),
            'FLPAD': lambda x: np.fliplr(np.rot90(x)),           
        }[self.name]


class Grid:
    def __init__(self, array: np.ndarray, *, 
                 idx: Optional[int] = None,
                 program_id: Optional[str] = None,
                 task_id: Optional[str] = None,
                 dataset: Optional[str] = None,
                 color_perm: Optional[str] = None,
                 transform: Optional[str] = None,
                 is_test: bool = False,
                 is_input: bool = True):
        self.array = np.asarray(array)
        self.idx = idx
        self.program_id = program_id
        self.task_id = task_id
        self.dataset = dataset
        self.color_perm = color_perm
        self.transform = transform
        self.is_test = is_test
        self.is_input = is_input
        
    @property
    def shape(self):
        return self.array.shape
    
    def __repr__(self):
        prefix = 'Test' if self.is_test else 'Train'
        io_type = 'Input' if self.is_input else 'Output'
        if all(x is not None for x in [self.program_id, self.dataset, self.task_id, self.idx]):
            return f"{self.program_id} : {self.dataset}/{self.task_id}/{prefix}/{self.idx}/{self.color_perm}/{self.transform}/{io_type}"
        return f"Grid(shape={self.shape})"
    
    def flatten(self, max_size: Optional[int] = None, pad_value: int = -1, eos_value: Optional[int] = None):
        """Flatten the grid array into a 1D array, optionally adding EOS markers and padding.

        Args:
            max_size (Optional[int]): Maximum length of the flattened array. If None, returns simple flattened array.
            pad_value (int): Value to use for padding when max_size is specified. Default is -1.
            eos_value (Optional[int]): Value to use for end-of-sequence markers. If None, no EOS markers are added.

        Returns:
            tuple: (flattened_array, position_indices)
                - flattened_array: 1D array with the flattened grid, and optionally EOS markers and padding
                - position_indices: Array of shape (N, 2) containing [row, col] indices for each position,
                  with [-1, -1] for padding and EOS positions
        """
        height, width = self.array.shape
        
        # Create position indices for the original grid
        row_indices, col_indices = np.indices((height, width))
        positions = np.stack([row_indices, col_indices], axis=-1)
        
        # Case 1: Simple flatten (original behavior)
        if max_size is None and eos_value is None:
            return self.array.flatten(), positions.reshape(-1, 2)

        # Calculate total size needed (with EOS markers if specified)
        total_size = height * (width + 1) if eos_value is not None else height * width
        
        # If max_size specified, validate it
        if max_size is not None and total_size > max_size:
            raise ValueError(f"Required size ({total_size}) exceeds max_size ({max_size})")
        
        # Create result arrays (with padding if max_size specified)
        result_size = max_size if max_size is not None else total_size
        result = np.full(result_size, pad_value, dtype=self.array.dtype)
        pos_result = np.full((result_size, 2), -1, dtype=np.int32)
        
        # Handle data placement
        if eos_value is not None:
            indices = np.arange(total_size).reshape(height, width + 1)
            # Place values and their positions
            result[indices[:, :-1].ravel()] = self.array.ravel()
            result[indices[:, -1]] = eos_value
            # Place position indices
            pos_result[indices[:, :-1].ravel()] = positions.reshape(-1, 2)
            # Add EOS positions: [row, width] for each row
            eos_positions = np.stack([np.arange(height), np.full(height, width)], axis=-1)
            pos_result[indices[:, -1]] = eos_positions
        else:
            result[:total_size] = self.array.ravel()
            pos_result[:total_size] = positions.reshape(-1, 2)
            
        return result, pos_result
    
    def tolist(self):
        return self.array.tolist()
    
    @staticmethod
    def from_array(array, **kwargs):
        return Grid(array, **kwargs)
    
    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.array, other.array)
    
    def __array__(self, dtype=None, copy=False):
        """Convert grid to numpy array with proper dtype and copy handling."""
        arr = self.array
        if dtype:
            arr = arr.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr
    
    def clone(self):
        """Create a deep copy of the grid."""
        return Grid(
            self.array.copy(),
            idx=self.idx,
            program_id=self.program_id,
            task_id=self.task_id,
            dataset=self.dataset,
            color_perm=self.color_perm,
            transform=self.transform,
            is_test=self.is_test,
            is_input=self.is_input
        )

    def permute(self, color_perm: ColorPermutation = ColorPermutation.RAND, arr_transform: Optional[ArrayTransform] = None, in_place: bool = False):
        """Apply color permutation and array transformation to the grid.
        
        Args:
            color_perm: Color permutation to apply. Defaults to a random permutation. If None, a random permutation except RAND is chosen.
            arr_transform: Array transformation to apply. If None, a random transformation is chosen.
            in_place: If True, modify this grid instance. If False, return a new grid.
        """
        try:
            if color_perm is None:
                cps = list(ColorPermutation)[:-1]  # Exclude the random permutation
                color_perm = random.choice(cps)
            else:
                assert isinstance(color_perm, ColorPermutation), "color_perm should be an instance of ColorPermutation"

            if arr_transform is None:
                ats = list(ArrayTransform)
                arr_transform = random.choice(ats)
            else:
                assert isinstance(arr_transform, ArrayTransform), "arr_transform should be an instance of ArrayTransform"

            # Try again if the identity transformation is selected with the identity color permutation
            if color_perm == ColorPermutation.CPID and arr_transform == ArrayTransform.IDENT:
                return self.permute()

            array = color_perm.transform(self.array)
            array = arr_transform.transform(array)

            if in_place:
                self.array = array
                self.color_perm = color_perm.name
                self.transform = arr_transform.name
                return self

            return Grid(
                array,
                idx=self.idx,
                program_id=self.program_id,
                task_id=self.task_id,
                dataset=self.dataset,
                color_perm=color_perm.name,
                transform=arr_transform.name,
                is_test=self.is_test,
                is_input=self.is_input
            )
        except Exception as e:
            # Log all relevant attributes for debugging
            logger.error(f"Error during permutation: {e}")
            logger.error(f"Grid attributes: idx={self.idx}, program_id={self.program_id}, task_id={self.task_id}, "
                         f"dataset={self.dataset}, color_perm={self.color_perm}, transform={self.transform}, "
                         f"is_test={self.is_test}, is_input={self.is_input}, array_shape={self.array.shape}")
            raise

    def project(self, new_height: int = 32, new_width: int = 32, pad_value: int = -1):
        """Project the grid array to a new height and width, padding with pad_value if necessary.

        Args:
            new_height (int): The desired height of the new array. Default is 32.
            new_width (int): The desired width of the new array. Default is 32.
            pad_value (int): The value to use for padding. Default is -1.

        Returns:
            np.ndarray: The projected array with the specified dimensions.
        """
        current_height, current_width = self.array.shape
        if current_height > new_height or current_width > new_width:
            raise ValueError("New dimensions must be greater than or equal to current dimensions.")

        # Calculate padding amounts
        pad_height = new_height - current_height
        pad_width = new_width - current_width

        # Pad the array
        padded_array = np.pad(
            self.array,
            pad_width=((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=pad_value
        )

        return padded_array

class Example:
    def __init__(self, idx: int, input: np.array, output: np.array, program_id: str, task_id: str, dataset: str, color_perm: str, transform: str, is_test=False):
        self.idx = idx    
        self.program_id = program_id
        self.task_id = task_id
        self.dataset = dataset
        self.color_perm = color_perm
        self.transform = transform
        self.is_test = is_test
        self._complexity = None
        self._is_original = color_perm == ColorPermutation.CPID.name and transform == ArrayTransform.IDENT.name
        self._original_input = None
        self._original_output = None
        
        # Create Grid objects with full metadata
        self.input = Grid(
            input,
            idx=idx,
            program_id=program_id,
            task_id=task_id,
            dataset=dataset,
            color_perm=color_perm,
            transform=transform,
            is_test=is_test,
            is_input=True
        )
        self.output = Grid(
            output,
            idx=idx,
            program_id=program_id,
            task_id=task_id,
            dataset=dataset,
            color_perm=color_perm,
            transform=transform,
            is_test=is_test,
            is_input=False
        )
    
    @property
    def is_original(self):
        return self._is_original
    
    def __repr__(self):
        prefix = 'Test' if self.is_test else 'Train'
        return f'{self.program_id} : {self.dataset}/{self.task_id}/{prefix}/{self.idx}/{self.color_perm}/{self.transform}'

    def to_dict(self):
        return {
            "idx": self.idx,
            "input": self.input.tolist(),
            "output": self.output.tolist(),
            "program_id": self.program_id,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "color_perm": self.color_perm,
            "transform": self.transform,
            "is_test": self.is_test
        }

    @staticmethod
    def from_dict(example_dict):
        return Example(
            idx=example_dict['idx'],
            input=np.array(example_dict['input']),
            output=np.array(example_dict['output']),
            program_id=example_dict['program_id'],
            task_id=example_dict['task_id'],
            dataset=example_dict['dataset'],
            color_perm=example_dict['color_perm'],
            transform=example_dict['transform'],
            is_test=example_dict['is_test']
        )

    def compute_complexity(self):
        size = max(len(self.input.array), len(self.output.array))/(30*30)
        scale = max(len(self.input.array)/len(self.output.array), len(self.output.array)/len(self.input.array))/(30*30)
        hist_inp, _ = np.histogram(self.input.array.flatten(), bins=np.arange(11), density=True)
        hist_out, _ = np.histogram(self.output.array.flatten(), bins=np.arange(11), density=True)
        color_var = np.sqrt(np.square(hist_inp - hist_out).sum())
        complexity = size*4 + scale*2 + color_var
        complexity = np.log(complexity + 1)
        return complexity

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = self.compute_complexity()
        return self._complexity

    def clone(self):
        """Create a deep copy of the example. But it is made as not original example."""
        assert self._is_original, "Cannot clone an permuted example."

        cloned = copy.deepcopy(self)
        cloned._is_original = False
        cloned._original_input = self.input.clone()
        cloned._original_output = self.output.clone()
        return cloned

    def permute(self, color_perm: Optional[ColorPermutation] = None, arr_transform: Optional[ArrayTransform] = None):
        assert not self._is_original, "Cannot transform an original example. Please clone first."

        if color_perm is None:
            cps = list(ColorPermutation)
            color_perm = random.choice(cps)
        else:
            assert isinstance(color_perm, ColorPermutation), "color_perm should be an instance of ColorPermutation"

        if arr_transform is None:
            ats = list(ArrayTransform)
            arr_transform = random.choice(ats)
        else:
            assert isinstance(arr_transform, ArrayTransform), "arr_transform should be an instance of ArrayTransform"

        # Try again if the identity transformation is selected with the identity color permutation
        if color_perm == ColorPermutation.CPID and arr_transform == ArrayTransform.IDENT:
            return self.permute()

        self.input = self.input.permute(color_perm, arr_transform)
        self.output = self.output.permute(color_perm, arr_transform)
        self.color_perm = color_perm.name
        self.transform = arr_transform.name
        return self


class ArcTask:
    def __init__(self, id, prog_id, train: List[Example], test: List[Example], dataset=None):
        self.id = id
        self.prog_id = prog_id
        self.dataset = dataset
        self.train = train
        self.test = test
        self._complexity = None

    def __repr__(self):
        return f'ArcTask(id={self.id}, prog={self.prog_id}, dataset={self.dataset})'

    def to_dict(self):
        result = {
            "id": self.id,
            "prog_id": self.prog_id,
            "dataset": self.dataset,
            "train": [e.to_dict() for e in self.train],
            "test": [e.to_dict() for e in self.test]
        }
        return result
    
    @staticmethod
    def from_dict(task_dict):
        return ArcTask(id=task_dict['id'],
                       prog_id=task_dict['prog_id'],
                       train=task_dict['train'],
                       test=task_dict['test'],
                       dataset=task_dict.get('dataset'))

    def compute_complexity(self):
        complexity = np.max([e.complexity for e in self.train + self.test])
        return complexity

    @property
    def complexity(self):
        if self._complexity is None:
            self._complexity = self.compute_complexity()
        return self._complexity

def load_task(task_json, task_id, prog_id, dataset, inverse=False):
    """Load a single task from JSON data."""
    train = [
        Example(
            idx=idx,
            input=np.array(ex['output']) if inverse else np.array(ex['input']),
            output=np.array(ex['input']) if inverse else np.array(ex['output']),
            program_id=prog_id,
            task_id=task_id,
            dataset=dataset,
            color_perm=ColorPermutation.CPID.name,
            transform=ArrayTransform.IDENT.name,
            is_test=False
        ) for idx, ex in enumerate(task_json['train'])
    ]
    
    test = [
        Example(
            idx=idx,
            input=np.array(ex['output']) if inverse else np.array(ex['input']),
            output=np.array(ex['input']) if inverse else np.array(ex['output']),
            program_id=prog_id,
            task_id=task_id,
            dataset=dataset,
            color_perm=ColorPermutation.CPID.name,
            transform=ArrayTransform.IDENT.name,
            is_test=True
        ) for idx, ex in enumerate(task_json['test'])
    ]
    
    return ArcTask(
        id=task_id,
        prog_id=prog_id,
        train=train,
        test=test,
        dataset=dataset
    )

def process_jsonl_file(file_path, name, prog_prefix, identical_task_per_folder, inverse=False):
    """Process a single JSONL file."""
    tasks = []
    with file_path.open('r') as file:
        for line_num, line in enumerate(file):
            task_json = json.loads(line.strip())
            
            # Create task ID and program ID
            task_id = f"{name}_{file_path.stem}_{line_num}"
            prog_id = file_path.parent.stem if identical_task_per_folder else f"{file_path.stem}_{line_num}"
            if prog_prefix:
                prog_id = prog_prefix + prog_id
            
            tasks.append(load_task(task_json, task_id, prog_id, name, inverse))
    return tasks

def process_json_file(file_path, name, prog_prefix, identical_task_per_folder, inverse=False):
    """Process a single JSON file."""
    task_json = json.load(file_path.open('r'))
    task_id = f"{name}_{file_path.stem}"
    prog_id = file_path.parent.stem if identical_task_per_folder else file_path.stem
    if prog_prefix:
        prog_id = prog_prefix + prog_id
    return [load_task(task_json, task_id, prog_id, name, inverse)]

class ArcTasksLoader:
    def __init__(self, name: str, path: str, prog_prefix='', identical_task_per_folder=False, inverse=False, has_jsonlines=False):
        self.name = name
        self.path = Path(path)
        self.prog_prefix = prog_prefix
        self.inverse = inverse
        self.identical_task_per_folder = identical_task_per_folder
        self._tasks = None  # Initialize _tasks
        self.has_jsonlines = has_jsonlines

    @property
    def tasks(self):
        if not self._tasks:
            self.load()
        return self._tasks
    
    @property
    def train_examples(self):
        return [example for task in self.tasks for example in task.train]
    
    @property
    def test_examples(self):
        return [example for task in self.tasks for example in task.test]
        
    @property
    def train_grids(self):
        return [grid for example in self.train_examples for grid in (example.input, example.output)]
    
    @property
    def test_grids(self):
        return [grid for example in self.test_examples for grid in (example.input, example.output)]

    def stats(self):
        logger.info(f"\n\nDataset: {self.name}")
        logger.info(f"Inverse Tasks: {self.inverse}")
        logger.info(f"Number of programs: {len(set([task.prog_id for task in self.tasks]))}")
        logger.info(f"Number of tasks: {len(self.tasks)}")
        logger.info(f"Number of train examples: {len(self.train_examples)}")
        logger.info(f"Number of test examples: {len(self.test_examples)}")
        logger.info(f"Average train examples per task: {len(self.train_examples)/len(self.tasks)}")
        logger.info(f"Average test examples per task: {len(self.test_examples)/len(self.tasks)}")

        
    def load(self):
        """Load tasks from either JSON or JSONL files in parallel."""
        # Determine file pattern and assert files exist
        file_pattern = "**/*.jsonl" if self.has_jsonlines else "**/*.json"
        files = sorted([f for f in Path(self.path).glob(file_pattern)])
        assert len(files) > 0, f"No {file_pattern} files found in {self.path}"
        
        # Choose processing function based on file type
        process_func = process_jsonl_file if self.has_jsonlines else process_json_file
        
        # Process files in parallel
        tasks = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = list(tqdm(
                executor.map(
                    partial(process_func, inverse=self.inverse),
                    files,
                    [self.name] * len(files),
                    [self.prog_prefix] * len(files),
                    [self.identical_task_per_folder] * len(files)
                ),
                total=len(files),
                desc=f"Loading {self.name} {'JSONL' if self.has_jsonlines else 'JSON'} files"
            ))
            # Flatten the list of lists
            tasks = [task for sublist in futures for task in sublist]
                
        self._tasks = tasks
        logger.info(f"Loaded {len(self._tasks)} tasks for {self.name}.")
        return self._tasks


    def __len__(self):
        return len(self.tasks)
    
    def get_inverse_loader(self, suffix='_INV', separate_prog=False):
        assert not self.inverse, "Cannot get inverse loader for an inverse loader"

        return ArcTasksLoader(
                    name=f"{self.name}{suffix}", 
                    path=self.path, 
                    prog_prefix=f'INV_{self.prog_prefix}' if separate_prog else self.prog_prefix,
                    identical_task_per_folder=self.identical_task_per_folder, 
                    inverse=True)
    


ARC_1D = ArcTasksLoader(name='ARC_1D', path='data/arc_dataset_collection/dataset/1D-ARC/data')
BARC_GP4OM_OM = ArcTasksLoader(name='BARC_GP4OM_OM', path='data/barc_tasks/data/100k_gpt4o-mini_generated_problems', has_jsonlines=True)
BARC_GP4_OM = ArcTasksLoader(name='BARC_GP4_OM', path='data/barc_tasks/data/100k-gpt4-description-gpt4omini-code_generated_problems', has_jsonlines=True)
BARC_GP4O_OM = ArcTasksLoader(name='BARC_GP4O_OM', path='data/barc_tasks/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems_data_100k', has_jsonlines=True)
BARC_GP4O_OM_SUG = ArcTasksLoader(name='BARC_GP4O_OM_SUG', path='data/barc_tasks/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems_data_suggestfunction_100k', has_jsonlines=True)
ARC_COMMUNITY = ArcTasksLoader(name='ARC_COMMUNITY', path='data/arc_dataset_collection/dataset/arc-community/data')
ARC_CONCEPT = ArcTasksLoader(name='ARC_CONCEPT', path='data/arc_dataset_collection/dataset/ConceptARC/data')
ARC_DBIGHAM = ArcTasksLoader(name='ARC_DBIGHAM', path='data/arc_dataset_collection/dataset/dbigham/data')
ARC_DIVA = ArcTasksLoader(name='ARC_DIVA', path='data/arc_dataset_collection/dataset/arc-dataset-diva/data', identical_task_per_folder=False)
ARC_MINI = ArcTasksLoader(name='ARC_MINI', path='data/arc_dataset_collection/dataset/Mini-ARC/data')
ARC_NOSOUND = ArcTasksLoader(name='ARC_NOSOUND', path='data/arc_dataset_collection/dataset/nosound/data')
ARC_PQA = ArcTasksLoader(name='ARC_PQA', path='data/arc_dataset_collection/dataset/PQA/data', identical_task_per_folder=False)
ARC_REARC_EASY = ArcTasksLoader(name='ARC_REARC_EASY', path='data/arc_dataset_collection/dataset/RE-ARC/data/easy', prog_prefix='REARCEASY')
ARC_REARC_HARD = ArcTasksLoader(name='ARC_REARC_HARD', path='data/arc_dataset_collection/dataset/RE-ARC/data/hard', prog_prefix='REARCHARD')
ARC_SEQUENCE = ArcTasksLoader(name='ARC_SEQUENCE', path='data/arc_dataset_collection/dataset/Sequence_ARC/data', prog_prefix='SEQ')
ARC_SORTOF = ArcTasksLoader(name='ARC_SORTOF', path='data/arc_dataset_collection/dataset/Sort-of-ARC/data')
ARC_SYNTH_RIDDLES = ArcTasksLoader(name='ARC_SYNTH_RIDDLES', path='data/arc_dataset_collection/dataset/synth_riddles/data')
ARC_TAMA = ArcTasksLoader(name='ARC_TAMA', path='data/arc_dataset_collection/dataset/arc-dataset-tama/data')
ARC_IPARC = ArcTasksLoader(name='ARC_IPARC', path='data/arc_dataset_collection/dataset/IPARC/data')

ARCAGI1_TRAIN = ArcTasksLoader(name='ARCAGI1_TRAIN', path='data/arc_dataset_collection/dataset/ARC-AGI-1/data/training')
ARCAGI1_EVAL = ArcTasksLoader(name='ARCAGI1_EVAL', path='data/arc_dataset_collection/dataset/ARC-AGI-1/data/evaluation')
ARCAGI2_TRAIN = ArcTasksLoader(name='ARCAGI2_TRAIN', path='data/arc_dataset_collection/dataset/ARC-AGI-2/data/training')
ARCAGI2_EVAL = ArcTasksLoader(name='ARCAGI2_EVAL', path='data/arc_dataset_collection/dataset/ARC-AGI-2/data/evaluation')

TRAIN_GRID_LOADERS = [
    BARC_GP4OM_OM, BARC_GP4_OM, BARC_GP4O_OM, BARC_GP4O_OM_SUG, 
    ARC_1D, ARC_COMMUNITY, ARC_CONCEPT, ARC_DBIGHAM,
    ARC_DIVA, ARC_MINI, ARC_NOSOUND, ARC_PQA,
    ARC_REARC_EASY, ARC_REARC_HARD, ARC_SEQUENCE, ARC_SORTOF,
    ARC_SYNTH_RIDDLES, ARC_TAMA, ARC_IPARC,
    ARCAGI1_TRAIN, ARCAGI1_EVAL, ARCAGI2_TRAIN
]

ARC_TRAIN = [ARCAGI1_TRAIN, ARCAGI2_TRAIN, ARCAGI1_EVAL]

VAL_GRID_LOADERS = [ARCAGI2_EVAL]



class DatasetType(str, Enum):
    ALL = "all"
    TRAIN = "train"  # Current train collection
    ARC_TRAIN = "arc_train"
    VAL = "val"    # Current val collection
    ARC_1D = "arc_1d"
    BARC_GP4OM_OM = "barc_gp4om_om"
    BARC_GP4_OM = "barc_gp4_om"
    BARC_GP4O_OM = "barc_gp4o_om"
    BARC_GP4O_OM_SUG = "barc_gp4o_om_sug"
    ARC_COMMUNITY = "arc_community"
    ARC_CONCEPT = "arc_concept"
    ARC_DBIGHAM = "arc_dbigham"
    ARC_DIVA = "arc_diva"
    ARC_MINI = "arc_mini"
    ARC_NOSOUND = "arc_nosound"
    ARC_PQA = "arc_pqa"
    ARC_REARC_EASY = "arc_rearc_easy"
    ARC_REARC_HARD = "arc_rearc_hard"
    ARC_SEQUENCE = "arc_sequence"
    ARC_SORTOF = "arc_sortof"
    ARC_SYNTH_RIDDLES = "arc_synth_riddles"
    ARC_TAMA = "arc_tama"
    ARC_IPARC = "arc_iparc"
    ARCAGI1_TRAIN = "arcagi1_train"
    ARCAGI1_EVAL = "arcagi1_eval"
    ARCAGI2_TRAIN = "arcagi2_train"
    ARCAGI2_EVAL = "arcagi2_eval"

# Map enum values to their corresponding loaders
DATASET_LOADERS = {
    DatasetType.ALL: TRAIN_GRID_LOADERS + VAL_GRID_LOADERS,
    DatasetType.TRAIN: TRAIN_GRID_LOADERS,
    DatasetType.ARC_TRAIN: ARC_TRAIN,
    DatasetType.VAL: VAL_GRID_LOADERS,
    DatasetType.ARC_1D: [ARC_1D],
    DatasetType.BARC_GP4OM_OM: [BARC_GP4OM_OM],
    DatasetType.BARC_GP4_OM: [BARC_GP4_OM],
    DatasetType.BARC_GP4O_OM: [BARC_GP4O_OM],
    DatasetType.BARC_GP4O_OM_SUG: [BARC_GP4O_OM_SUG],
    DatasetType.ARC_COMMUNITY: [ARC_COMMUNITY],
    DatasetType.ARC_CONCEPT: [ARC_CONCEPT],
    DatasetType.ARC_DBIGHAM: [ARC_DBIGHAM],
    DatasetType.ARC_DIVA: [ARC_DIVA],
    DatasetType.ARC_MINI: [ARC_MINI],
    DatasetType.ARC_NOSOUND: [ARC_NOSOUND],
    DatasetType.ARC_PQA: [ARC_PQA],
    DatasetType.ARC_REARC_EASY: [ARC_REARC_EASY],
    DatasetType.ARC_REARC_HARD: [ARC_REARC_HARD],
    DatasetType.ARC_SEQUENCE: [ARC_SEQUENCE],
    DatasetType.ARC_SORTOF: [ARC_SORTOF],
    DatasetType.ARC_SYNTH_RIDDLES: [ARC_SYNTH_RIDDLES],
    DatasetType.ARC_TAMA: [ARC_TAMA],
    DatasetType.ARC_IPARC: [ARC_IPARC],
    DatasetType.ARCAGI1_TRAIN: [ARCAGI1_TRAIN],
    DatasetType.ARCAGI1_EVAL: [ARCAGI1_EVAL],
    DatasetType.ARCAGI2_TRAIN: [ARCAGI2_TRAIN],
    DatasetType.ARCAGI2_EVAL: [ARCAGI2_EVAL],
}

# %%
