from collections import defaultdict
import logging
import pickle
from pathlib import Path
import random
from typing import List

from .data import ArcTasksLoader
from .utils import hash_string

logger = logging.getLogger(__name__)

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
        logger.info(f"Loading {len(self.loaders)} loaders")
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

    @property
    def train_grids(self):
        """Returns a list of all input and output grids from training examples."""
        if not self._train:
            self.load()
        all_grids = []
        for examples in self._train.values():
            for example in examples:
                all_grids.extend([example.input, example.output])
        return all_grids
    
    @property
    def test_grids(self):
        """Returns a list of all input and output grids from test examples."""
        if not self._test:
            self.load()
        all_grids = []
        for examples in self._test.values():
            for example in examples:
                all_grids.extend([example.input, example.output])
        return all_grids

# Loader instances
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

def get_task_loaders(*, train=True, evl=True, aux=True, inv=False, separate_inv_prog=True):
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

def load_dataset(*, task_loaders, max_height=30, max_width=30, min_test=1, min_train=1, max_test=100, max_train=100, no_cache=False):
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