import numpy as np
from unittest.mock import patch, MagicMock
import pytest
from pips.grid_dataset import GridDataset, process_grid_loader, combined_dtype
from pips.data import Grid

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

def test_grid_dataset_initialization(mock_load_grid_loaders):
    # Initialize the GridDataset
    dataset = GridDataset(loaders=[])

    # Check the length of the dataset
    assert len(dataset) == 1

    # Retrieve the first item
    grid = dataset[0]

    # Check that the item is a Grid object
    assert isinstance(grid, Grid)

    # Check the attributes of the Grid object
    assert np.array_equal(grid.array, np.array([[1, 2], [3, 4]]))
    assert grid.idx == 0
    assert grid.program_id == 'prog_id'
    assert grid.task_id == 'task_id'
    assert grid.dataset == 'dataset'
    assert grid.color_perm == 'color_perm'
    assert grid.transform == 'transform'
    assert grid.is_test == np.bool_(False)
    assert grid.is_input == np.bool_(True) 