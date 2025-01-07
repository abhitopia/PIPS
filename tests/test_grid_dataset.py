import numpy as np
from unittest.mock import patch, MagicMock
from pips.grid_dataset import process_grid_loader, combined_dtype

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