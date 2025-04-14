import numpy as np
import pytest

from pips.data import Example, ArcTask, ColorPermutation, ArrayTransform, Grid
from pips.task_dataset_1 import ArcTrainingDataset

# Test fixtures
@pytest.fixture
def basic_example():
    return Example(
        idx=0,
        input=np.array([[0, 1], [2, 3]]),
        output=np.array([[3, 2], [1, 0]]),
        program_id="test_prog",
        task_id="test_task",
        dataset="test_dataset",
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name,
        is_test=False
    )

def test_arc_training_dataset(basic_example):
    # Create a simple loader with one task
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[basic_example],
        test=[basic_example],
        dataset="test_dataset"
    )
    
    class MockLoader:
        def __init__(self, tasks):
            self.tasks = tasks
    
    mock_loader = MockLoader([task])
    
    # Create dataset
    dataset = ArcTrainingDataset([mock_loader])
    dataset.load()
    
    assert len(dataset.train) == 1
    assert len(dataset.test) == 1
    assert list(dataset.train.keys())[0] == "test_prog"

def test_dataset_augmentation(basic_example):
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[basic_example],
        test=[basic_example],
        dataset="test_dataset"
    )
    
    class MockLoader:
        def __init__(self, tasks):
            self.tasks = tasks
    
    mock_loader = MockLoader([task])
    dataset = ArcTrainingDataset([mock_loader])
    dataset.load()
    
    # Test augmentation
    dataset.augment(min_train=5, max_train=10, min_test=3, max_test=5)
    
    assert len(dataset.train["test_prog"]) >= 5
    assert len(dataset.test["test_prog"]) >= 3

def test_arc_training_dataset_filtering():
    # Create example with large dimensions
    valid_input = np.zeros((20, 20))
    valid_output = np.zeros((20, 20))
    large_input = np.zeros((50, 50))
    large_output = np.zeros((50, 50))
    large_example = Example(
        idx=0,
        input=large_input,
        output=large_output,
        program_id="test_prog",
        task_id="test_task",
        dataset="test_dataset",
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name,
        is_test=False
    )
    valid_example = Example(
        idx=1,
        input=valid_input,
        output=valid_output,
        program_id="test_prog",
        task_id="test_task",
        dataset="test_dataset",
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name,
        is_test=False
    )
    
    # Create task with large example
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[large_example, valid_example],
        test=[large_example, valid_example],
        dataset="test_dataset"
    )
    
    class MockLoader:
        def __init__(self, tasks):
            self.tasks = tasks
    
    mock_loader = MockLoader([task])
    dataset = ArcTrainingDataset([mock_loader])
    dataset.load()

    assert len(dataset.train_examples) == 2
    assert len(dataset.test_examples) == 2
    assert len(dataset.train) == 1
    assert len(dataset.test) == 1
    
    # Test filtering
    dataset.filter(max_height=30, max_width=30)

    assert len(dataset.train_examples) == 1
    assert len(dataset.test_examples) == 1
    assert len(dataset.train) == 1
    assert len(dataset.test) == 1

def test_dataset_train_test_grids():
    # Create a simple dataset with known grids
    input_array = np.array([[1, 2], [3, 4]])
    output_array = np.array([[4, 3], [2, 1]])
    example = Example(
        idx=0,
        input=input_array,
        output=output_array,
        program_id="test_prog",
        task_id="test_task",
        dataset="test_dataset",
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name,
        is_test=False
    )
    
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[example],
        test=[example],
        dataset="test_dataset"
    )
    
    class MockLoader:
        def __init__(self, tasks):
            self.tasks = tasks
    
    mock_loader = MockLoader([task])
    dataset = ArcTrainingDataset([mock_loader])
    dataset.load()
    
    # Test train_grids
    train_grids = dataset.train_grids
    assert len(train_grids) == 2  # One input and one output grid
    assert isinstance(train_grids[0], Grid)
    assert isinstance(train_grids[1], Grid)
    
    # Test test_grids
    test_grids = dataset.test_grids
    assert len(test_grids) == 2  # One input and one output grid
    assert isinstance(test_grids[0], Grid)
    assert isinstance(test_grids[1], Grid) 