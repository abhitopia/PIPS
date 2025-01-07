import logging
import concurrent.futures
from pips.data import ArcTasksLoader, Grid
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

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

GRID_LOADERS = [
    BARC_GP4OM_OM, BARC_GP4_OM, BARC_GP4O_OM, BARC_GP4O_OM_SUG, 
    ARC_1D, ARC_COMMUNITY, ARC_CONCEPT, ARC_DBIGHAM,
    ARC_DIVA, ARC_EVAL, ARC_MINI, ARC_NOSOUND, ARC_PQA,
    ARC_REARC_EASY, ARC_REARC_HARD, ARC_SEQUENCE, ARC_SORTOF,
    ARC_SYNTH_RIDDLES, ARC_TAMA, ARC_TRAIN, ARC_IPARC
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
    combined_data = np.zeros(len(grids), dtype=combined_dtype)

    for i, grid in tqdm(enumerate(grids), total=len(grids), desc=f"Processing grids for {loader.name}"):
        if grid.array.shape[0] > 30 or grid.array.shape[1] > 30:
            continue
        # Initialize a 30x30 grid with -1
        grid_data = np.full((30, 30), -1, dtype=np.int8)
        
        # Copy the smaller grid into the larger grid
        grid_data[:grid.array.shape[0], :grid.array.shape[1]] = grid.array

        combined_data[i] = (
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

    np.save(output_file, combined_data)

def load_grid_loaders(loaders, cache_dir=Path(__file__).resolve().parent.parent / '.cache'):
    cache_dir.mkdir(parents=True, exist_ok=True)

    unified_file = cache_dir / 'grid_data.npy'

    if not unified_file.exists():
        output_files = {}
        for loader in loaders:
            output_file = cache_dir / f"{loader.name}_grid_data.npy"
            output_files[loader.name] = output_file

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_grid_loader, loader, output_files[loader.name]): loader for loader in loaders}

            for future in concurrent.futures.as_completed(futures):
                loader = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {loader.name}: {e}")

        all_data = []

        for loader_name, output_file in output_files.items():
            assert output_file.exists(), f"Output file for {loader_name} does not exist"
            data = np.load(output_file, mmap_mode='r')
            all_data.append(data)

        all_data = np.concatenate(all_data, axis=0)

        logger.info(f"Saving grid data to cache...")
        np.save(unified_file, all_data)
    else:
        logger.info(f"Loading grid data from cache...")
        all_data = np.load(unified_file, mmap_mode='r')

    return all_data

class GridDataset(Dataset):
    def __init__(self, loaders, cache_dir=Path(__file__).resolve().parent.parent / '.cache'):
        # Load the data using the existing function
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

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format to just show the message
    )

    dataset = GridDataset(GRID_LOADERS)
    print(len(dataset))

    for i in tqdm(range(len(dataset))):
        grid = dataset[i]
        print(grid)

    #%%
    # ARC_1D.load()
    # BARC_GP4_OM.load()
 
    # BARC_GP4_OM.stats()
    # %%
