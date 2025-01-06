#%%
import copy
from enum import Enum, auto
import json
import logging
from pathlib import Path
import random
from typing import List, Optional
import numpy as np

#%%

logger = logging.getLogger(__name__)

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
    
    def flatten(self):
        return self.array.flatten()
    
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

    def permute(self, color_perm: ColorPermutation, arr_transform: ArrayTransform, in_place: bool = False):
        """Apply color permutation and array transformation to the grid.
        
        Args:
            color_perm: Color permutation to apply
            arr_transform: Array transformation to apply
            in_place: If True, modify this grid instance. If False, return a new grid.
        """
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

class ArcTasksLoader:
    def __init__(self, name: str, path: str, prog_prefix='', identical_task_per_folder=False, inverse=False):
        base_path = Path(__file__).resolve().parent.parent    
        self.path = base_path / Path(path)
        assert self.path.exists(), f'Path does not exist: {self.path}'
        self.name = name
        self.prog_prefix = prog_prefix
        self.inverse = inverse
        self._tasks = None
        self._train_examples = None
        self._test_examples = None
        self.identical_task_per_folder = identical_task_per_folder

    def json_files(self):
        return 
    
    @property
    def tasks(self):
        if not self._tasks:
            self.load_from_json_files()
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

    def load_examples(self, task_id, prog_id, examples_json, is_test=False) -> List[Example]:
        examples = []
        for idx, example in enumerate(examples_json):
            examples.append(
                    Example(
                    idx=idx,
                    input=np.asarray(example['input']) if not self.inverse else np.asarray(example['output']),
                    output=np.asarray(example['output']) if not self.inverse else np.asarray(example['input']),
                    program_id=prog_id,
                    task_id=task_id,
                    dataset=self.name,
                    color_perm=ColorPermutation.CPID.name, # This is the identity transformation
                    transform=ArrayTransform.IDENT.name,
                    is_test=is_test
                    )
                )
        return examples

    def stats(self):
        logger.info(f"\n\nDataset: {self.name}")
        logger.info(f"Inverse Tasks: {self.inverse}")
        logger.info(f"Number of programs: {len(set([task.prog_id for task in self.tasks]))}")
        logger.info(f"Number of tasks: {len(self.tasks)}")
        logger.info(f"Number of train examples: {len(self.train_examples)}")
        logger.info(f"Number of test examples: {len(self.test_examples)}")
        logger.info(f"Average train examples per task: {len(self.train_examples)/len(self.tasks)}")
        logger.info(f"Average test examples per task: {len(self.test_examples)/len(self.tasks)}")
    
    def load_from_json_files(self) -> None:
        json_files = sorted([json for json in Path(self.path).glob("**/*.json")])
        tasks = []
        train_examples = []
        test_examples = []
        for f in json_files:
            task_json = json.load(f.open('r'))
            task_id = self.name + "_" + f.stem

            if self.identical_task_per_folder:
                prog_id = f.parent.stem
            else:
                prog_id = f.stem

            if self.prog_prefix:
                prog_id = self.prog_prefix + prog_id

            train = self.load_examples(task_id, prog_id, task_json['train'], is_test=False)
            test = self.load_examples(task_id, prog_id, task_json['test'], is_test=True)
            train_examples.extend(train)
            test_examples.extend(test)
            task = ArcTask(
                    id=task_id,
                    prog_id=prog_id,
                    train=train,
                    test=test,
                    dataset=self.name
                )  
            tasks.append(task)

        self._tasks = tasks

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
    

