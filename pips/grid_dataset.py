import logging
import concurrent.futures
from pips.data import ArcTasksLoader, Grid
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from typing import NamedTuple, List, Dict
import hashlib

logger = logging.getLogger(__name__)

ARC_1D = ArcTasksLoader(name='ARC_1D', path='data/arc_dataset_collection/dataset/1D-ARC/data')
BARC_GP4OM_OM = ArcTasksLoader(name='BARC_GP4OM_OM', path='data/barc_tasks/data/100k_gpt4o-mini_generated_problems', has_jsonlines=True)
BARC_GP4_OM = ArcTasksLoader(name='BARC_GP4_OM', path='data/barc_tasks/data/100k-gpt4-description-gpt4omini-code_generated_problems', has_jsonlines=True)
BARC_GP4O_OM = ArcTasksLoader(name='BARC_GP4O_OM', path='data/barc_tasks/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems_data_100k', has_jsonlines=True)
BARC_GP4O_OM_SUG = ArcTasksLoader(name='BARC_GP4O_OM_SUG', path='data/barc_tasks/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems_data_suggestfunction_100k', has_jsonlines=True)
ARC_COMMUNITY = ArcTasksLoader(name='ARC_COMMUNITY', path='data/arc_dataset_collection/dataset/arc-community/data')
ARC_CONCEPT = ArcTasksLoader(name='ARC_CONCEPT', path='data/arc_dataset_collection/dataset/ConceptARC/data')
ARC_DBIGHAM = ArcTasksLoader(name='ARC_DBIGHAM', path='data/arc_dataset_collection/dataset/dbigham/data')
ARC_DIVA = ArcTasksLoader(name='ARC_DIVA', path='data/arc_dataset_collection/dataset/arc-dataset-diva/data', identical_task_per_folder=False)
ARC_EVAL = ArcTasksLoader(name='ARC_EVAL', path='data/arc_dataset_collection/dataset/ARC/data/evaluation')
ARC_MINI = ArcTasksLoader(name='ARC_MINI', path='data/arc_dataset_collection/dataset/Mini-ARC/data')
ARC_NOSOUND = ArcTasksLoader(name='ARC_NOSOUND', path='data/arc_dataset_collection/dataset/nosound/data')
ARC_PQA = ArcTasksLoader(name='ARC_PQA', path='data/arc_dataset_collection/dataset/PQA/data', identical_task_per_folder=False)
ARC_REARC_EASY = ArcTasksLoader(name='ARC_REARC_EASY', path='data/arc_dataset_collection/dataset/RE-ARC/data/easy', prog_prefix='REARCEASY')
ARC_REARC_HARD = ArcTasksLoader(name='ARC_REARC_HARD', path='data/arc_dataset_collection/dataset/RE-ARC/data/hard', prog_prefix='REARCHARD')
ARC_SEQUENCE = ArcTasksLoader(name='ARC_SEQUENCE', path='data/arc_dataset_collection/dataset/Sequence_ARC/data', prog_prefix='SEQ')
ARC_SORTOF = ArcTasksLoader(name='ARC_SORTOF', path='data/arc_dataset_collection/dataset/Sort-of-ARC/data')
ARC_SYNTH_RIDDLES = ArcTasksLoader(name='ARC_SYNTH_RIDDLES', path='data/arc_dataset_collection/dataset/synth_riddles/data')
ARC_TAMA = ArcTasksLoader(name='ARC_TAMA', path='data/arc_dataset_collection/dataset/arc-dataset-tama/data')
ARC_TRAIN = ArcTasksLoader(name='ARC_TRAIN', path='data/arc_dataset_collection/dataset/ARC/data/training')
ARC_IPARC = ArcTasksLoader(name='ARC_IPARC', path='data/arc_dataset_collection/dataset/IPARC/data')

TRAIN_GRID_LOADERS = [
    BARC_GP4OM_OM, BARC_GP4_OM, BARC_GP4O_OM, BARC_GP4O_OM_SUG, 
    ARC_1D, ARC_COMMUNITY, ARC_CONCEPT, ARC_DBIGHAM,
    ARC_DIVA, ARC_MINI, ARC_NOSOUND, ARC_PQA,
    ARC_REARC_EASY, ARC_REARC_HARD, ARC_SEQUENCE, ARC_SORTOF,
    ARC_SYNTH_RIDDLES, ARC_TAMA, ARC_TRAIN, ARC_IPARC
]

VAL_GRID_LOADERS = [
    ARC_EVAL
]

# Define a structured data type for the combined grid and attributes
combined_dtype = np.dtype([
    ('grid', (np.int8, (30, 30))),  # Assuming grid size is 30x30
    ('idx', np.int32),
    ('program_id', 'U50'),  # Assuming max length of 50
    ('task_id', 'U50'),
    ('dataset', 'U50'),
    ('color_perm', 'U10'),
    ('transform', 'U10'),
    ('is_test', np.bool_),
    ('is_input', np.bool_),
    ('original_shape', (np.int32, 2))  # Assuming shape is a tuple of two integers
])

def process_grid_loader(loader, output_file):
    """Function to load tasks for a specific loader and save to a single NPY file."""
    if output_file.exists():
        logger.info(f"Cache exists for {loader.name}. Skipping.")
        return

    grids = loader.train_grids + loader.test_grids

    # Log the number of grids loaded
    logger.info(f"Loaded {len(grids)} grids for {loader.name}")

    # Check if grids are empty
    if not grids:
        logger.warning(f"No grids found for {loader.name}. Check the data source.")
        return

    combined_data = np.zeros(len(grids), dtype=combined_dtype)

    valid_grid_count = 0
    for i, grid in tqdm(enumerate(grids), total=len(grids), desc=f"Processing grids for {loader.name}"):
        if grid.array.shape[0] > 30 or grid.array.shape[1] > 30:
            logger.debug(f"Skipping grid {grid.idx} with shape {grid.array.shape} for {loader.name}")
            continue

        # Initialize a 30x30 grid with -1
        grid_data = np.full((30, 30), -1, dtype=np.int8)
        
        # Copy the smaller grid into the larger grid
        grid_data[:grid.array.shape[0], :grid.array.shape[1]] = grid.array

        combined_data[valid_grid_count] = (
            grid_data,
            grid.idx,
            grid.program_id,
            grid.task_id,
            grid.dataset,
            grid.color_perm,
            grid.transform,
            grid.is_test,
            grid.is_input,
            grid.array.shape
        )
        valid_grid_count += 1

    if valid_grid_count == 0:
        logger.warning(f"No valid grids processed for {loader.name}. Check grid dimensions and filtering criteria.")
        return

    # Resize combined_data to only include valid grids
    combined_data = combined_data[:valid_grid_count]

    np.save(output_file, combined_data)
    logger.info(f"Saved {valid_grid_count} valid grids to {output_file} for {loader.name}")

def load_grid_loaders(loaders, cache_dir=Path(__file__).resolve().parent.parent / '.cache'):
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sort loaders by name
    sorted_loaders = sorted(loaders, key=lambda loader: loader.name)

    # Create a hash from the sorted loader names
    loader_names = ''.join(loader.name for loader in sorted_loaders)
    unified_file_hash = hashlib.md5(loader_names.encode()).hexdigest()
    unified_file = cache_dir / f'grid_data_{unified_file_hash}.npy'

    if not unified_file.exists():
        output_files = {}
        for loader in sorted_loaders:
            output_file = cache_dir / f"{loader.name}_grid_data.npy"
            output_files[loader.name] = output_file

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_grid_loader, loader, output_files[loader.name]): loader for loader in sorted_loaders}

            for future in concurrent.futures.as_completed(futures):
                loader = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {loader.name}: {e}")

        all_data = []

        logger.info(f"Saving grid data to cache...")
        for loader_name, output_file in output_files.items():
            assert output_file.exists(), f"Output file for {loader_name} does not exist"
            data = np.load(output_file, mmap_mode='r')
            all_data.append(data)

        all_data = np.concatenate(all_data, axis=0)

        np.save(unified_file, all_data)
    else:
        logger.info(f"Loading grid data from cache...")
        all_data = np.load(unified_file, mmap_mode='r')

    return all_data

# Define the NamedTuple for the collate function output
class GRID_INPUT(NamedTuple):
    grids: torch.Tensor  # (B, S) where S is the flattened size of the projected grid
    attributes: List[Dict[str, any]]

class GridDataset(Dataset):
    def __init__(self, train: bool = True, cache_dir=Path(__file__).resolve().parent.parent / '.cache'):
        # Load the data using the existing function

        loaders = TRAIN_GRID_LOADERS if train else VAL_GRID_LOADERS
        self.data = load_grid_loaders(loaders, cache_dir)

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the sample at the given index
        sample = self.data[idx]
        
        # Extract the original shape
        original_shape = sample['original_shape']
        
        # Reshape the grid to its original dimensions
        grid_array = sample['grid'][:original_shape[0], :original_shape[1]]
        
        # Create a Grid object using the attributes from the sample
        grid = Grid(
            array=grid_array,
            idx=sample['idx'],
            program_id=sample['program_id'],
            task_id=sample['task_id'],
            dataset=sample['dataset'],
            color_perm=sample['color_perm'],
            transform=sample['transform'],
            is_test=sample['is_test'],
            is_input=sample['is_input']
        )
        
        return grid

    @staticmethod
    def collate_fn(batch, pad_value=-1, device=torch.device('cpu'), permute=False, project_size=(32, 32)) -> GRID_INPUT:
        """Collate function to process a batch of Grids.

        Args:
            batch (list of Grid): The batch of Grid objects.
            pad_value (int): The value to use for padding. Default is -1.
            device (str or torch.device): The device to move the tensors to. Default is 'cpu'.
            permute (bool): Whether to permute the grid before projection. Default is False.
            project_size (tuple): The desired size for grid projection. Default is (32, 32).

        Returns:
            GRID_INPUT: A named tuple containing the projected grids and their attributes.
        """
        projected_grids = []
        attributes = []

        for grid in batch:
            # Optionally permute the grid using the grid.permute method
            if permute:
                grid = grid.permute()

            # Project each grid to the specified size
            projected_array = grid.project(new_height=project_size[0], new_width=project_size[1], pad_value=pad_value)
            projected_grids.append(projected_array)

            # Collect attributes and convert numpy types to native Python types
            attributes.append({
                'idx': int(grid.idx),
                'program_id': str(grid.program_id),
                'task_id': str(grid.task_id),
                'dataset': str(grid.dataset),
                'color_perm': str(grid.color_perm),
                'transform': str(grid.transform),
                'is_test': bool(grid.is_test),
                'is_input': bool(grid.is_input)
            })

        # Convert the list of numpy arrays to a single numpy array before converting to a tensor
        projected_grids = np.array(projected_grids)
        projected_grids = torch.tensor(projected_grids, dtype=torch.long).to(device, non_blocking=True)

        # Flatten the projected grids to shape BSx(project_size[0] * project_size[1])
        projected_grids = projected_grids.view(projected_grids.size(0), -1)

        return GRID_INPUT(grids=projected_grids, attributes=attributes)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format to just show the message
    )

    dataset = GridDataset(train=True)
    print(f"Total number of samples in dataset: {len(dataset)}")

    # Create a DataLoader with batch size 32, pad_value 15, and random shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: GridDataset.collate_fn(x, pad_value=15, device='cpu', permute=True),
        drop_last=False
    )

    # Iterate over the DataLoader
    for batch in dataloader:
        print(f"Batch grids shape: {batch.grids.shape}")  # Should be (32, 32, 32)
        print(f"Batch attributes: {batch.attributes}")
        break  # Just process the first batch for demonstration

