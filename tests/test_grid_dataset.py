import numpy as np
from unittest.mock import patch, MagicMock
import pytest
from pips.grid_dataset import GridDataset, process_grid_loader, combined_dtype, GRID_INPUT, TRAIN_GRID_LOADERS, VAL_GRID_LOADERS
from pips.data import Grid
import torch
from pathlib import Path

def test_process_grid_loader():
    # Mock a loader with fake grids
    mock_loader = MagicMock()
    mock_loader.name = "MOCK_LOADER"
    mock_loader.train_grids = [MagicMock(array=np.array([[1, 2], [3, 4]]))]
    mock_loader.test_grids = [MagicMock(array=np.array([[5, 6], [7, 8]]))]

    # Add necessary attributes to mock grids
    for grid in mock_loader.train_grids + mock_loader.test_grids:
        grid.idx = 0
        grid.program_id = "prog_id"
        grid.task_id = "task_id"
        grid.dataset = "dataset"
        grid.color_perm = "color_perm"
        grid.transform = "transform"
        grid.is_test = False
        grid.is_input = True

    # Mock the output file path
    mock_output_file = MagicMock()
    mock_output_file.exists.return_value = False

    # Run the process_grid_loader function
    with patch('numpy.save') as mock_save:
        process_grid_loader(mock_loader, mock_output_file)

        # Check that numpy.save was called once
        mock_save.assert_called_once()

        # Check the data passed to numpy.save
        saved_data = mock_save.call_args[0][1]
        assert saved_data.dtype == combined_dtype
        assert saved_data.shape[0] == 2  # Two grids: one train, one test

        # Check the grid data
        assert np.array_equal(saved_data[0]['grid'][:2, :2], np.array([[1, 2], [3, 4]]))
        assert np.array_equal(saved_data[1]['grid'][:2, :2], np.array([[5, 6], [7, 8]]))

# Mock the load_grid_loaders function to return a controlled dataset
@pytest.fixture
def mock_load_grid_loaders():
    with patch('pips.grid_dataset.load_grid_loaders') as mock_loader:
        # Create a grid of shape (30, 30) with padding
        grid_data = np.full((30, 30), -1, dtype=np.int8)
        grid_data[:2, :2] = np.array([[1, 2], [3, 4]])  # Fill in the actual data

        mock_loader.return_value = np.array([
            (
                grid_data,  # grid
                0,  # idx
                'prog_id',  # program_id
                'task_id',  # task_id
                'dataset',  # dataset
                'color_perm',  # color_perm
                'transform',  # transform
                np.bool_(False),  # is_test
                np.bool_(True),  # is_input
                (2, 2)  # original_shape
            )
        ], dtype=combined_dtype)
        yield mock_loader

@patch('pips.grid_dataset.load_grid_loaders')
def test_grid_dataset_initialization(mock_load_grid_loaders):
    # Mock the return value of load_grid_loaders
    mock_load_grid_loaders.return_value = []

    # Define the cache directory
    cache_dir = Path(__file__).resolve().parent.parent / '.cache'

    # Initialize the GridDataset for training
    dataset = GridDataset(train=True)
    assert len(dataset) == 0

    # Initialize the GridDataset for validation
    dataset = GridDataset(train=False)
    assert len(dataset) == 0

    # Update assertion to include verbose parameter
    mock_load_grid_loaders.assert_any_call(TRAIN_GRID_LOADERS, cache_dir, verbose=True)

def test_collate_fn():
    # Create mock Grid objects
    mock_grids = [
        MagicMock(spec=Grid, array=np.array([[1, 2], [3, 4]]), idx=0, program_id='prog_id_0', task_id='task_id_0',
                  dataset='dataset_0', color_perm='color_perm_0', transform='transform_0', is_test=False, is_input=True),
        MagicMock(spec=Grid, array=np.array([[5, 6], [7, 8]]), idx=1, program_id='prog_id_1', task_id='task_id_1',
                  dataset='dataset_1', color_perm='color_perm_1', transform='transform_1', is_test=True, is_input=False)
    ]

    # Mock the flatten method to return a padded array
    for grid in mock_grids:
        grid.flatten.return_value = np.array([*grid.array.flatten(), *[-1] * (1024 - 4)])

    # Call the collate function
    result = GridDataset.collate_fn(mock_grids, pad_value=-1, device='cpu', max_size=1024)

    # Check the type of the result
    assert isinstance(result, GRID_INPUT)

    # Check the shape of the grids tensor
    assert result.grids.shape == (2, 1024)

    # Check that the attributes are correct
    assert len(result.attributes) == 2
    assert result.attributes[0]['idx'] == 0
    assert result.attributes[0]['program_id'] == 'prog_id_0'
    assert result.attributes[0]['task_id'] == 'task_id_0'
    assert result.attributes[0]['dataset'] == 'dataset_0'
    assert result.attributes[0]['color_perm'] == 'color_perm_0'
    assert result.attributes[0]['transform'] == 'transform_0'
    assert result.attributes[0]['is_test'] is False
    assert result.attributes[0]['is_input'] is True

    assert result.attributes[1]['idx'] == 1
    assert result.attributes[1]['program_id'] == 'prog_id_1'
    assert result.attributes[1]['task_id'] == 'task_id_1'
    assert result.attributes[1]['dataset'] == 'dataset_1'
    assert result.attributes[1]['color_perm'] == 'color_perm_1'
    assert result.attributes[1]['transform'] == 'transform_1'
    assert result.attributes[1]['is_test'] is True
    assert result.attributes[1]['is_input'] is False

def test_process_grid_loader_empty_grids():
    # Mock a loader with no grids
    mock_loader = MagicMock()
    mock_loader.name = "EMPTY_LOADER"
    mock_loader.train_grids = []
    mock_loader.test_grids = []

    # Mock the output file path
    mock_output_file = MagicMock()
    mock_output_file.exists.return_value = False

    # Run the process_grid_loader function
    with patch('pips.grid_dataset.logger') as mock_logger:
        process_grid_loader(mock_loader, mock_output_file)

        # Check that a warning was logged
        mock_logger.warning.assert_called_with(f"No grids found for {mock_loader.name}. Check the data source.")

def test_process_grid_loader_valid_grids():
    # Mock a loader with valid grids
    mock_loader = MagicMock()
    mock_loader.name = "VALID_LOADER"
    mock_loader.train_grids = [MagicMock(array=np.array([[1, 2], [3, 4]]))]
    mock_loader.test_grids = [MagicMock(array=np.array([[5, 6], [7, 8]]))]

    # Add necessary attributes to mock grids
    for grid in mock_loader.train_grids + mock_loader.test_grids:
        grid.idx = 0
        grid.program_id = "prog_id"
        grid.task_id = "task_id"
        grid.dataset = "dataset"
        grid.color_perm = "color_perm"
        grid.transform = "transform"
        grid.is_test = False
        grid.is_input = True

    # Mock the output file path
    mock_output_file = MagicMock()
    mock_output_file.exists.return_value = False

    # Run the process_grid_loader function
    with patch('numpy.save') as mock_save, patch('pips.grid_dataset.logger') as mock_logger:
        process_grid_loader(mock_loader, mock_output_file)

        # Check that numpy.save was called once
        mock_save.assert_called_once()

        # Check that the info log was called for saving grids
        mock_logger.info.assert_any_call(f"Loaded 2 grids for {mock_loader.name}")
        mock_logger.info.assert_any_call(f"Saved 2 valid grids to {mock_output_file} for {mock_loader.name}")

def test_collate_fn_different_sizes():
    # Create mock Grid objects with different sizes
    mock_grids = [
        MagicMock(spec=Grid, array=np.array([[1, 2], [3, 4]]), idx=0, program_id='prog_id_0', task_id='task_id_0',
                  dataset='dataset_0', color_perm='color_perm_0', transform='transform_0', is_test=False, is_input=True),
        MagicMock(spec=Grid, array=np.array([[5, 6], [7, 8]]), idx=1, program_id='prog_id_1', task_id='task_id_1',
                  dataset='dataset_1', color_perm='color_perm_1', transform='transform_1', is_test=True, is_input=False)
    ]

    # Mock the flatten method to return arrays padded to max_size=512
    for grid in mock_grids:
        grid.flatten.return_value = np.array([*grid.array.flatten(), *[-1] * (512 - 4)])

    # Call the collate function with a different max_size
    result = GridDataset.collate_fn(mock_grids, pad_value=-1, device='cpu', max_size=512)

    # Check the shape of the grids tensor matches the specified max_size
    assert result.grids.shape == (2, 512)

    # Check that the attributes are correct
    assert len(result.attributes) == 2
    assert result.attributes[0]['idx'] == 0
    assert result.attributes[0]['program_id'] == 'prog_id_0'
    assert result.attributes[0]['task_id'] == 'task_id_0'
    assert result.attributes[0]['dataset'] == 'dataset_0'
    assert result.attributes[0]['color_perm'] == 'color_perm_0'
    assert result.attributes[0]['transform'] == 'transform_0'
    assert result.attributes[0]['is_test'] is False
    assert result.attributes[0]['is_input'] is True

    assert result.attributes[1]['idx'] == 1
    assert result.attributes[1]['program_id'] == 'prog_id_1'
    assert result.attributes[1]['task_id'] == 'task_id_1'
    assert result.attributes[1]['dataset'] == 'dataset_1'
    assert result.attributes[1]['color_perm'] == 'color_perm_1'
    assert result.attributes[1]['transform'] == 'transform_1'
    assert result.attributes[1]['is_test'] is True
    assert result.attributes[1]['is_input'] is False

# Add more tests as needed... 