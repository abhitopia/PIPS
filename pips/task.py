#%%
from collections import defaultdict
import copy
from enum import Enum, auto
import json
import logging
from pathlib import Path
import pickle
import random
from typing import List, Optional
import numpy as np

from .utils import hash_string
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

    @property
    def transform(self):
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


class Example:
    def __init__(self, idx: int, input: np.array, output: np.array, program_id: str, task_id: str, dataset: str, color_perm: str, transform: str, is_test=False):
        self.idx = idx    
        self.input = input
        self.output = output
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

    @property
    def is_original(self):
        return self._is_original
    
    def __repr__(self):
        prefix = 'Test' if self.is_test else 'Train'
        return f'{self.program_id} : {self.dataset}/{self.task_id}/{prefix}/{self.idx}/{self.color_perm}/{self.transform}'

    def to_dict(self):
        return {
            "idx": self.idx,
            "input": np.asarray(self.input),
            "output": np.asarray(self.output),
            "program_id": self.program_id,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "color_perm": self.color_perm,
            "transform": self.transform,
            "is_test": self.is_test
        }

    @staticmethod
    def from_dict(example_dict):
        return Example(idx=example_dict['idx'],
                       input=example_dict['input'].tolist(),
                       output=example_dict['output'].tolist(),
                       program_id=example_dict['program_id'],
                       task_id=example_dict['task_id'],
                       dataset=example_dict['dataset'],
                       color_perm=example_dict['color_perm'],
                       transform=example_dict['transform'],
                       is_test=example_dict['is_test'])

    def compute_complexity(self):
        size = max(len(self.input), len(self.output))/(30*30)
        scale = max(len(self.input)/len(self.output), len(self.output)/len(self.input))/(30*30)
        hist_inp, _ = np.histogram(self.input.flatten(), bins=np.arange(11), density=True)
        hist_out, _ = np.histogram(self.output.flatten(), bins=np.arange(11), density=True)
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
        cloned._original_input = np.copy(self.input)
        cloned._original_output = np.copy(self.output)
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

        input = color_perm.transform(self.input)
        output = color_perm.transform(self.output)
        input = arr_transform.transform(input)
        output = arr_transform.transform(output)

        self.input = input
        self.output = output
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
            self.load()
        return self._tasks
    
    @property
    def train(self):
        return [example for task in self.tasks for example in task.train]
    
    @property
    def test(self):
        return [example for task in self.tasks for example in task.test]

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
        logger.info(f"Number of train examples: {len(self.train)}")
        logger.info(f"Number of test examples: {len(self.test)}")
        logger.info(f"Average train examples per task: {len(self.train)/len(self.tasks)}")
        logger.info(f"Average test examples per task: {len(self.test)/len(self.tasks)}")
    
    def load(self) -> None:
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


class ArcTrainingDataset:
    def __init__(self, loaders: List[ArcTasksLoader]):
        self.loaders = loaders
        self._train = defaultdict(list)
        self._test = defaultdict(list)

    @property
    def tasks(self):
        if not self._tasks:
            self.load()
        return self._tasks

    def load(self):
        for loader in self.loaders:
            for task in loader.tasks:
                self._train[task.prog_id].extend(task.train)
                self._test[task.prog_id].extend(task.test)
        assert len(self._train) == len(self._test), 'Number of programs in train and test should be the same'

    def stats(self):
        num_progs = len(self.train)
        num_train = sum([len(v) for v in self._train.values()])
        num_test = sum([len(v) for v in self._test.values()])
        logger.info(f"\n\nTraining Data Stats:")
        logger.info(f"Number of programs: {num_progs}")
        logger.info(f"Number of train examples: {num_train}")
        logger.info(f"Number of test examples: {num_test}")
        logger.info(f"Average train examples per task: {num_train/num_progs}")
        logger.info(f"Average test examples per task: {num_test/num_progs}")

        color_permutations = defaultdict(int)
        transformations = defaultdict(int)
        dataset_counts = defaultdict(int)
        for prog_id in self.train.keys():
            train_examples = self.train[prog_id]
            test_examples = self.test[prog_id]
            for example in train_examples:
                color_permutations[example.color_perm] += 1
                transformations[example.transform] += 1
                dataset_counts[example.dataset] += 1

            for example in test_examples:
                color_permutations[example.color_perm] += 1
                transformations[example.transform] += 1
                dataset_counts[example.dataset] += 1

        logger.info("Color Permutations:")
        for k, v in sorted(color_permutations.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"\t{k}: {v}")

        logger.info("Transformations:")
        for k, v in sorted(transformations.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"\t{k}: {v}")

        total_datasets = sum(dataset_counts.values())
        logger.info("Datasets:")
        for k, v in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"\t{k}: {v} ({v/total_datasets:.2f})")


        # Number of program per dataset
        dataset_progs = defaultdict(set)
        for prog_id in self.train.keys():
            for example in self.train[prog_id]:
                dataset_progs[example.dataset].add(prog_id)
        
        logger.info("Programs per dataset:")
        for k, v in sorted(dataset_progs.items(), key=lambda x: len(x[1]), reverse=True):
            logger.info(f"\t{k}: {len(v)}")

    @property
    def train(self):
        if not self._train:
            self.load()
        return self._train
    
    @property
    def train_examples(self):
        if not self._train:
            self.load()
        return sum([es for es in self._train.values()], [])
    
    @property
    def test(self):
        if not self._test:
            self.load()
        return self._test
    
    @property
    def test_examples(self):
        if not self._test:
            self.load()
        return sum([es for es in self._test.values()], [])

    def __len__(self):
        return len(self._train)
    
    def filter(self, max_height=45, max_width=45):
        new_train = defaultdict(list)
        new_test = defaultdict(list)

        def is_valid(example, max_height, max_width):
            inp_height, inp_width = example.input.shape
            out_height, out_width = example.output.shape
            return inp_height <= max_height and inp_width <= max_width and out_height <= max_height and out_width <= max_width

        for prog_id in self.train.keys():
            train_exampled_kept = []
            test_examples_kept = []
            for example in self.train[prog_id]:
                if is_valid(example, max_height, max_width):
                    train_exampled_kept.append(example)

            for example in self.test[prog_id]:
                if is_valid(example, max_height, max_width):
                    test_examples_kept.append(example)

            if len(train_exampled_kept) > 0 and len(test_examples_kept) > 0:
                new_train[prog_id] = train_exampled_kept
                new_test[prog_id] = test_examples_kept

        self._train = new_train
        self._test = new_test

    def augment(self, min_train=50, max_train=None, min_test=3, max_test=None):
        # Set max_train and max_test to min values if not provided
        if max_train is None:
            max_train = min_train
        if max_test is None:
            max_test = min_test

        # Assert that min values are less than or equal to max values
        assert min_train <= max_train, "min_train should be less than or equal to max_train"
        assert min_test <= max_test, "min_test should be less than or equal to max_test"

        new_train = defaultdict(list)
        new_test = defaultdict(list)

        for prog_id in self.train.keys():
            # Get current test examples
            test_examples = self.test[prog_id]
            num_test = len(test_examples)

            # Adjust test examples to be within min_test and max_test
            if num_test > max_test:
                # Reduce test examples
                keep_indices = random.sample(range(num_test), max_test)
                kept_test_examples = [test_examples[i] for i in keep_indices]
                discarded_test_examples = [test_examples[i] for i in range(num_test) if i not in keep_indices]
            else:
                kept_test_examples = test_examples.copy()
                discarded_test_examples = []
                # Augment test examples if below min_test
                if num_test < min_test:
                    num_to_augment = min_test - num_test
                    augment_examples = [random.choice(kept_test_examples) for _ in range(num_to_augment)]
                    augmented_tests = [example.clone().permute() for example in augment_examples]
                    kept_test_examples.extend(augmented_tests)

            new_test[prog_id] = kept_test_examples

            # Prepare train examples by adding discarded test examples first
            train_examples = self.train[prog_id] + discarded_test_examples
            num_train = len(train_examples)

            # If train examples are less than min_train, augment after using discarded test examples
            if num_train < min_train:
                num_to_augment = min_train - num_train
                augment_examples = [random.choice(train_examples) for _ in range(num_to_augment)]
                augmented_trains = [example.clone().permute() for example in augment_examples]
                train_examples.extend(augmented_trains)
                num_train = len(train_examples)

            # Ensure train_examples do not exceed max_train
            if num_train > max_train:
                train_examples = random.sample(train_examples, max_train)

            new_train[prog_id] = train_examples

        self._train = new_train
        self._test = new_test


    
#%%
ARC_1D = ArcTasksLoader(name='ARC_1D', path='data/arc_dataset_collection/dataset/1D-ARC/data', identical_task_per_folder=True)
ARC_COMMUNITY = ArcTasksLoader(name='ARC_COMMUNITY', path='data/arc_dataset_collection/dataset/arc-community/data')
ARC_CONCEPT = ArcTasksLoader(name='ARC_CONCEPT', path='data/arc_dataset_collection/dataset/ConceptARC/data')
ARC_DBIGHAM = ArcTasksLoader(name='ARC_DBIGHAM', path='data/arc_dataset_collection/dataset/dbigham/data')
ARC_DIVA = ArcTasksLoader(name='ARC_DIVA', path='data/arc_dataset_collection/dataset/arc-dataset-diva/data', identical_task_per_folder=True)
ARC_EVAL = ArcTasksLoader(name='ARC_EVAL', path='data/arc_dataset_collection/dataset/ARC/data/evaluation')
ARC_MINI = ArcTasksLoader(name='ARC_MINI', path='data/arc_dataset_collection/dataset/Mini-ARC/data')
ARC_NOSOUND = ArcTasksLoader(name='ARC_NOSOUND', path='data/arc_dataset_collection/dataset/nosound/data')
ARC_PQA = ArcTasksLoader(name='ARC_PQA', path='data/arc_dataset_collection/dataset/PQA/data', identical_task_per_folder=True)
ARC_REARC_EASY = ArcTasksLoader(name='ARC_REARC_EASY', path='data/arc_dataset_collection/dataset/RE-ARC/data/easy', prog_prefix='REARCEASY')
ARC_REARC_HARD = ArcTasksLoader(name='ARC_REARC_HARD', path='data/arc_dataset_collection/dataset/RE-ARC/data/hard', prog_prefix='REARCHARD')
ARC_SEQUENCE = ArcTasksLoader(name='ARC_SEQUENCE', path='data/arc_dataset_collection/dataset/Sequence_ARC/data', prog_prefix='SEQ')
ARC_SORTOF = ArcTasksLoader(name='ARC_SORTOF', path='data/arc_dataset_collection/dataset/Sort-of-ARC/data')
ARC_SYNTH = ArcTasksLoader(name='ARC_SYNTH', path='data/synthetic/', identical_task_per_folder=True)
ARC_SYNTHTASK = ArcTasksLoader(name='ARC_SYNTHTASK', path='data/arc_synth_tasks/data/synthetic_tasks')
ARC_SYNTH_RIDDLES = ArcTasksLoader(name='ARC_SYNTH_RIDDLES', path='data/arc_dataset_collection/dataset/synth_riddles/data')
ARC_TAMA = ArcTasksLoader(name='ARC_TAMA', path='data/arc_dataset_collection/dataset/arc-dataset-tama/data')
ARC_TRAIN = ArcTasksLoader(name='ARC_TRAIN', path='data/arc_dataset_collection/dataset/ARC/data/training')



train_collection = [
    ARC_TRAIN
]

eval_collection = [
    ARC_EVAL
]

aux_collection = [
    ARC_1D,
    ARC_COMMUNITY, 
    ARC_CONCEPT, 
    ARC_DBIGHAM, 
    ARC_DIVA, 
    ARC_MINI, 
    ARC_NOSOUND, 
    ARC_PQA, 
    ARC_REARC_EASY,
    ARC_REARC_HARD,
    ARC_SEQUENCE, 
    ARC_SORTOF,
    ARC_SYNTH,
    ARC_SYNTHTASK,
    ARC_SYNTH_RIDDLES, 
    ARC_TAMA,
]



def get_task_loaders(*, train=True, evl=True, aux=True, inv=False, separate_inv_prog=False):
    loaders = []
    if train:
        loaders.extend(train_collection)
        if inv:
            loaders.extend([loader.get_inverse_loader(separate_prog=separate_inv_prog) for loader in train_collection])
    if evl:
        loaders.extend(eval_collection)
        if inv:
            loaders.extend([loader.get_inverse_loader(separate_prog=separate_inv_prog) for loader in eval_collection])
    if aux:
        loaders.extend(aux_collection)
        if inv:
            loaders.extend([loader.get_inverse_loader(separate_prog=separate_inv_prog) for loader in aux_collection])
    return loaders


def load_dataset(*, task_loaders, max_height, max_width, min_test, min_train, max_test, max_train, no_cache=False):
    sorted_loaders = sorted(task_loaders, key=lambda x: x.name)
    loader_hash = hash_string("_".join([loader.name for loader in sorted_loaders]))

    base_path = Path(__file__).resolve().parent.parent / ".cache"
    hash_params = f"{loader_hash}_{max_height}_{max_width}_{min_test}_{max_test}_{max_train}_{min_train}"
    cache_file = base_path / f"{hash_string(hash_params)}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not no_cache:
        print(f"Loading dataset from cache: {cache_file}")
        dataset = pickle.load(cache_file.open("rb"))
    else:
        dataset = ArcTrainingDataset(loaders=sorted_loaders)
        dataset.filter(max_height=max_height, max_width=max_width)
        dataset.augment(min_test=min_test, max_test=max_test, max_train=max_train, min_train=min_train)
        pickle.dump(dataset, cache_file.open("wb"))
    return dataset

