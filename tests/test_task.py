import json
import numpy as np
import pytest
from pathlib import Path

from pips.task import (
    ColorPermutation,
    ArrayTransform,
    Example,
    ArcTask,
    ArcTasksLoader,
    ArcTrainingDataset,
    Grid
)

# Test fixtures
@pytest.fixture
def simple_input_array():
    return np.array([[0, 1], [2, 3]])

@pytest.fixture
def simple_output_array():
    return np.array([[3, 2], [1, 0]])

@pytest.fixture
def basic_example(simple_input_array, simple_output_array):
    return Example(
        idx=0,
        input=simple_input_array,
        output=simple_output_array,
        program_id="test_prog",
        task_id="test_task",
        dataset="test_dataset",
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name,
        is_test=False
    )


def test_all_color_permutations():
    # Test array with all possible colors 0-9
    test_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Expected results for each permutation
    expected_results = {
        'CPID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Identity
        'CP01': [7, 4, 5, 2, 8, 3, 0, 9, 6, 1],
        'CP02': [0, 9, 4, 5, 6, 8, 1, 3, 2, 7],
        'CP03': [7, 4, 1, 9, 6, 0, 8, 2, 5, 3],
        'CP04': [9, 6, 5, 7, 4, 0, 3, 8, 1, 2],
        'CP05': [1, 8, 0, 3, 9, 5, 6, 2, 7, 4],
        'CP06': [5, 3, 1, 9, 7, 6, 0, 2, 8, 4],
        'CP07': [1, 4, 3, 8, 7, 9, 6, 2, 5, 0],
        'CP08': [6, 0, 2, 1, 3, 4, 7, 8, 5, 9],
        'CP09': [2, 0, 3, 8, 4, 6, 1, 9, 5, 7]
    }
    
    # Test each permutation
    for perm in ColorPermutation:
        transform = perm.transform
        result = transform(test_array)
        expected = np.array(expected_results[perm.name])
        np.testing.assert_array_equal(
            result, 
            expected,
            err_msg=f"Failed for permutation {perm.name}"
        )

def test_color_permutation_2d():
    # Test 2D array
    test_array = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    
    # Test CP01 permutation on 2D array
    cp01_transform = ColorPermutation.CP01.transform
    result = cp01_transform(test_array)
    expected = np.array([
        [7, 4, 5],
        [2, 8, 3],
        [0, 9, 6]
    ])
    np.testing.assert_array_equal(result, expected)

def test_color_permutation_properties():
    test_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Test that applying a permutation twice doesn't return to original
    # (except for identity permutation)
    for perm in ColorPermutation:
        if perm != ColorPermutation.CPID:
            transform = perm.transform
            result = transform(transform(test_array))
            assert not np.array_equal(result, test_array), \
                f"Permutation {perm.name} is self-inverse when it shouldn't be"
    
    # Test that all permutations preserve the number of unique colors
    for perm in ColorPermutation:
        transform = perm.transform
        result = transform(test_array)
        assert len(np.unique(result)) == len(np.unique(test_array)), \
            f"Permutation {perm.name} changed the number of unique colors"

def test_color_permutation_edge_cases():

    # Test array with single value
    single_array = np.array([5])
    for perm in ColorPermutation:
        transform = perm.transform
        result = transform(single_array)
        assert result.shape == single_array.shape
    
    # Test array with values outside 0-9 range
    invalid_array = np.array([10, 11, -1])
    for perm in ColorPermutation:
        transform = perm.transform
        result = transform(invalid_array)
        expected = np.array([None if x not in range(10) else perm.value[x] for x in invalid_array])
        np.testing.assert_array_equal(result, expected)

# Test ArrayTransform
def test_array_transform():
    test_array = np.array([[1, 2], [3, 4]])
    
    # Test rotation 90
    rotated = ArrayTransform.RT090.transform(test_array)
    expected_rot90 = np.array([[2, 4], [1, 3]])
    np.testing.assert_array_equal(rotated, expected_rot90)
    
    # Test flip left-right
    flipped = ArrayTransform.FLPLR.transform(test_array)
    expected_flip = np.array([[2, 1], [4, 3]])
    np.testing.assert_array_equal(flipped, expected_flip)

# Test Example class
def test_example_creation(basic_example):
    assert basic_example.idx == 0
    assert basic_example.program_id == "test_prog"
    assert basic_example.is_original == True
    assert basic_example.is_test == False

def test_example_clone_and_permute(basic_example):
    # Test cloning
    cloned = basic_example.clone()
    assert cloned.is_original == False
    assert np.array_equal(cloned.input, basic_example.input)
    
    # Test permutation
    permuted = cloned.permute(
        color_perm=ColorPermutation.CP01,
        arr_transform=ArrayTransform.RT090
    )
    assert permuted.color_perm == ColorPermutation.CP01.name
    assert permuted.transform == ArrayTransform.RT090.name
    assert not np.array_equal(permuted.input, basic_example.input)

def test_example_complexity(basic_example):
    complexity = basic_example.complexity
    assert isinstance(complexity, float)
    assert complexity > 0

# Test ArcTask class
def test_arc_task(basic_example):
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[basic_example],
        test=[basic_example],
        dataset="test_dataset"
    )
    
    assert task.id == "test_task"
    assert len(task.train) == 1
    assert len(task.test) == 1
    assert task.complexity > 0

# Test ArcTasksLoader
def test_arc_tasks_loader(tmp_path):
    # Create a temporary test JSON file
    test_json = {
        "train": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[3, 2], [1, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[3, 2], [1, 0]]
            }
        ]
    }
    
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    test_file = test_dir / "test_task.json"
    
    import json
    with open(test_file, 'w') as f:
        json.dump(test_json, f)
    
    loader = ArcTasksLoader(
        name="test_dataset",
        path=str(test_dir),
        prog_prefix="test_"
    )
    
    loader.load()
    assert len(loader.tasks) == 1
    assert len(loader.train) == 1
    assert len(loader.test) == 1

# Test ArcTrainingDataset
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

# Add these new test functions after the existing ones

def test_all_array_transforms():
    # Test array with distinct values for clear transformation testing
    test_array = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    expected_results = {
        'IDENT': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'RT090': np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]]),
        'RT180': np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
        'RT270': np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
        'FLPLR': np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]]),
        'FLPUD': np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]]),
        'FLPDG': np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]), 
        'FLPAD': np.array([[9, 6, 3], [8, 5, 2], [7, 4, 1]])  
    }
    
    for transform in ArrayTransform:
        result = transform.transform(test_array)
        if transform.name not in expected_results:
            continue
        expected = expected_results[transform.name]
        np.testing.assert_array_equal(
            result,
            expected,
            err_msg=f"Failed for transform {transform.name}"
        )

def test_array_transform_properties():
    test_array = np.array([[1, 2], [3, 4]])
    
    # Test that RT090 applied 4 times returns to original
    result = test_array
    for _ in range(4):
        result = ArrayTransform.RT090.transform(result)
    np.testing.assert_array_equal(result, test_array)
    
    # Test that FLPLR and FLPUD are self-inverse
    for transform in [ArrayTransform.FLPLR, ArrayTransform.FLPUD]:
        result = transform.transform(transform.transform(test_array))
        np.testing.assert_array_equal(
            result,
            test_array,
            err_msg=f"Transform {transform.name} is not self-inverse"
        )

def test_array_transform_edge_cases():
    # Test 1x1 array
    single_element = np.array([[1]])
    for transform in ArrayTransform:
        result = transform.transform(single_element)
        np.testing.assert_array_equal(result, single_element)
    

def test_example_serialization(basic_example):
    # Test to_dict and from_dict
    example_dict = basic_example.to_dict()
    reconstructed = Example.from_dict(example_dict)
    
    assert reconstructed.idx == basic_example.idx
    assert reconstructed.program_id == basic_example.program_id
    assert reconstructed.task_id == basic_example.task_id
    assert reconstructed.dataset == basic_example.dataset
    assert reconstructed.color_perm == basic_example.color_perm
    assert reconstructed.transform == basic_example.transform
    assert reconstructed.is_test == basic_example.is_test
    np.testing.assert_array_equal(reconstructed.input, basic_example.input)
    np.testing.assert_array_equal(reconstructed.output, basic_example.output)

def test_arc_task_serialization(basic_example):
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[basic_example],
        test=[basic_example],
        dataset="test_dataset"
    )
    
    task_dict = task.to_dict()
    reconstructed = ArcTask.from_dict(task_dict)
    
    assert reconstructed.id == task.id
    assert reconstructed.prog_id == task.prog_id
    assert reconstructed.dataset == task.dataset
    assert len(reconstructed.train) == len(task.train)
    assert len(reconstructed.test) == len(task.test)

def test_arc_tasks_loader_inverse(tmp_path): # tmp_path is a pytest fixture
    # Test inverse loading functionality
    test_json = {
        "train": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[3, 2], [1, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[3, 2], [1, 0]]
            }
        ]
    }
    
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    test_file = test_dir / "test_task.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_json, f)
    
    # Test normal loading
    normal_loader = ArcTasksLoader(
        name="test_dataset",
        path=str(test_dir),
        inverse=False
    )
    normal_loader.load()
    
    # Test inverse loading
    inverse_loader = normal_loader.get_inverse_loader()
    inverse_loader.load()
    
    # Check that input and output are swapped in inverse loader
    normal_example = normal_loader.tasks[0].train[0]
    inverse_example = inverse_loader.tasks[0].train[0]
    
    np.testing.assert_array_equal(normal_example.input, inverse_example.output)
    np.testing.assert_array_equal(normal_example.output, inverse_example.input)

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

def test_dataset_complexity_calculation(basic_example):
    task = ArcTask(
        id="test_task",
        prog_id="test_prog",
        train=[basic_example],
        test=[basic_example],
        dataset="test_dataset"
    )
    
    # Test that complexity is calculated and cached
    assert task._complexity is None
    complexity = task.complexity
    assert isinstance(complexity, float)
    assert task._complexity is not None
    
    # Test that cached complexity is used
    cached_complexity = task.complexity
    assert cached_complexity == complexity

def test_grid_creation_and_properties():
    # Test basic grid creation
    array = np.array([[1, 2], [3, 4]])
    grid = Grid(array, 
                idx=0,
                program_id="test_prog",
                task_id="test_task",
                dataset="test_dataset",
                color_perm=ColorPermutation.CPID.name,
                transform=ArrayTransform.IDENT.name,
                is_test=False,
                is_input=True)
    
    # Test properties
    assert grid.shape == (2, 2)
    assert np.array_equal(grid.array, array)
    assert grid.idx == 0
    assert grid.program_id == "test_prog"
    assert grid.task_id == "test_task"
    assert grid.dataset == "test_dataset"
    assert grid.color_perm == ColorPermutation.CPID.name
    assert grid.transform == ArrayTransform.IDENT.name
    assert not grid.is_test
    assert grid.is_input

def test_grid_operations():
    array = np.array([[1, 2], [3, 4]])
    grid = Grid(array)
    
    # Test flatten
    assert np.array_equal(grid.flatten(), np.array([1, 2, 3, 4]))
    
    # Test tolist
    assert grid.tolist() == [[1, 2], [3, 4]]
    
    # Test equality
    grid2 = Grid(array.copy())
    assert grid == grid2
    assert grid != Grid(np.array([[4, 3], [2, 1]]))
    
    # Test array conversion
    assert np.array_equal(np.array(grid), array)

def test_grid_clone():
    array = np.array([[1, 2], [3, 4]])
    original = Grid(array,
                   idx=0,
                   program_id="test_prog",
                   task_id="test_task",
                   dataset="test_dataset",
                   color_perm=ColorPermutation.CPID.name,
                   transform=ArrayTransform.IDENT.name)
    
    cloned = original.clone()
    
    # Test that arrays are equal but separate
    assert np.array_equal(cloned.array, original.array)
    assert cloned.array is not original.array
    
    # Test that metadata is preserved
    assert cloned.idx == original.idx
    assert cloned.program_id == original.program_id
    assert cloned.task_id == original.task_id
    assert cloned.dataset == original.dataset
    assert cloned.color_perm == original.color_perm
    assert cloned.transform == original.transform

def test_grid_permute():
    array = np.array([[1, 2], [3, 4]])
    grid = Grid(array)
    
    # Test permutation without in-place modification
    permuted = grid.permute(ColorPermutation.CP01, ArrayTransform.RT090)
    assert permuted is not grid
    assert permuted.color_perm == ColorPermutation.CP01.name
    assert permuted.transform == ArrayTransform.RT090.name
    
    # Test in-place permutation
    original_array = grid.array.copy()
    grid.permute(ColorPermutation.CP01, ArrayTransform.RT090, in_place=True)
    assert grid.color_perm == ColorPermutation.CP01.name
    assert grid.transform == ArrayTransform.RT090.name
    assert not np.array_equal(grid.array, original_array)

def test_example_original_tracking(basic_example):
    # Test that original example is marked as original
    assert basic_example.is_original
    assert basic_example._original_input is None
    assert basic_example._original_output is None
    
    # Test that cloned example is not marked as original and stores original grids
    cloned = basic_example.clone()
    assert not cloned.is_original
    assert isinstance(cloned._original_input, Grid)
    assert isinstance(cloned._original_output, Grid)
    
    # Test that permutation maintains non-original status
    permuted = cloned.permute()
    assert not permuted.is_original
    assert isinstance(permuted._original_input, Grid)
    assert isinstance(permuted._original_output, Grid)

def test_example_permutation_restrictions(basic_example):
    # Test that original example cannot be permuted directly
    with pytest.raises(AssertionError):
        basic_example.permute()
    
    # Test that cloned example can be permuted
    cloned = basic_example.clone()
    try:
        cloned.permute()
    except AssertionError:
        pytest.fail("Permutation of cloned example raised AssertionError")

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