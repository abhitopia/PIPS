from typing import Union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Assuming Grid, Example, ArcTask are defined in pips.data
# If they are in the same directory, the import might need adjustment based on project structure
try:
    from .data import Grid, Example, ArcTask, ARCAGI1_EVAL
except ImportError:
    # Fallback for running script directly or different structure
    from pips.data import Grid, Example, ArcTask, ARCAGI1_EVAL


# Color map based on user provided selection, adding pad value -1
COLOR_MAP = {
    -1: '#FFFFFF', # white for padding
    0: '#000000',  # black
    1: '#0074D9',  # blue
    2: '#FF4136',  # red
    3: '#2ECC40',  # green
    4: '#FFDC00',  # yellow
    5: '#AAAAAA',  # grey
    6: '#F012BE',  # fuchsia
    7: '#FF851B',  # orange
    8: '#7FDBFF',  # teal
    9: '#870C25',  # brown
    15: '#FFFFFF',  # white
}

# Add the missing color from the original ARC definition (not in user selection but standard)
# 10 is often used as a background/canvas color in some ARC tasks, map it to white
# COLOR_MAP[10] = '#FFFFFF' 

# Prepare colormap and normalization for matplotlib
_sorted_keys = sorted(COLOR_MAP.keys())
_cmap = mcolors.ListedColormap([COLOR_MAP[k] for k in _sorted_keys])
_norm = mcolors.BoundaryNorm([k - 0.5 for k in _sorted_keys] + [_sorted_keys[-1] + 0.5], _cmap.N)


def extract_attributes(object: Union[Grid, Example, ArcTask]) -> str:
    """Extracts attributes from a Grid, Example, or ArcTask object and returns them as a formatted string."""
    parts = []
    parts.append(f"{object.dataset} | {object.program_id} | {object.task_id}")
    parts.append(f"{object.color_perm.name} | {object.transform.name}")

    if isinstance(object, (Example, Grid)):
        subparts = []
        test_train = f"{'Test' if object.is_test else 'Train'}"
        subparts.append(test_train)
        idx = f"{object.idx}"
        subparts.append(idx)
        if isinstance(object, Grid):
            input_output = f"{'Input' if object.is_input else 'Output'}"
            subparts.append(input_output)
        parts.append(" | ".join(subparts))
    return parts

def visualize_grid(grid: Grid, ax=None, title: str = None, show_gridlines: bool = True, show_attributes_in_title: bool = True):
    """Visualizes a single Grid object using matplotlib.

    Args:
        grid (Grid): The Grid object to visualize.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates new figure and axes. Defaults to None.
        title (str, optional): Base title for the plot. Defaults to None.
        show_gridlines (bool): Whether to show grid lines. Defaults to True.
        show_attributes_in_title (bool): If True, constructs a detailed title from grid attributes. Defaults to False.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        show_plot = True
    else:
        show_plot = False # Don't show if axes were provided

    array = grid.array
    rows, cols = array.shape

    ax.imshow(array, cmap=_cmap, norm=_norm, interpolation='nearest', extent=(-0.5, cols-0.5, rows-0.5, -0.5))
    
    # Set ticks for grid cells
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add gridlines if requested
    if show_gridlines:
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0) # Hide minor tick marks
    # Ensure aspect ratio is equal
    ax.set_aspect('equal', adjustable='box')

    # --- Title and Attribute Display --- 
    plot_title = title if title else "Grid" # Basic title
    ax.set_title(plot_title, fontsize=10)

    if show_attributes_in_title: # Renamed flag, but logic now places text below

        parts = extract_attributes(grid)
        attribute_text = "\n".join(parts)
        
        # Place attribute text below the plot using ax.text
        # Coordinates are relative to axes (0,0 is bottom-left, 1,1 is top-right)
        # x=0.5 -> horizontal center
        # y=-0.1 -> slightly below the bottom axis
        # ha='center', va='top' -> horizontally center, vertically align top of text block to y coordinate
        ax.text(0.5, -0.1, attribute_text, 
                ha='center', va='top', fontsize=8, 
                transform=ax.transAxes, wrap=True)

    # --- End Title and Attribute Display ---
    
    if show_plot:
        # Adjust bottom margin only if we created the figure AND added text
        if show_attributes_in_title:
             plt.subplots_adjust(bottom=0.25) # Make space for text below
        plt.show() 


def visualize_example(example: Example, title: str = None, show_attributes_in_title: bool = True):
    """Visualizes an Example object (input and output grids) using visualize_grid,
    showing detailed attributes for the example and its grids.

    Args:
        example (Example): The Example object to visualize.
        title (str, optional): Optional base title prefix for the overall plot. Defaults to None.
    """
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4)) # Old side-by-side layout
    fig, axes = plt.subplots(2, 1, figsize=(5, 9)) # Adjusted size slightly for potentially longer titles
    
    # Call visualize_grid telling it to show attributes in the title for input/output
    visualize_grid(example.input, ax=axes[0], title="Input", show_gridlines=True, show_attributes_in_title=False)
    visualize_grid(example.output, ax=axes[1], title="Output", show_gridlines=True, show_attributes_in_title=False)

    # Construct detailed suptitle with Example attributes
    parts = []
    if title: parts.append(title)
    if show_attributes_in_title:
        parts.extend(extract_attributes(example))

    sup_title = "\n".join(parts)
    fig.suptitle(sup_title, fontsize=11) 
    
    # Adjust layout 
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust top margin for suptitle
    plt.show()


def visualize_task(task: ArcTask):
    """Visualizes all examples (train and test) in an ArcTask within a single figure.

    Args:
        task (ArcTask): The task to visualize.
        task_index (int, optional): The index of the task for titling. Defaults to None.
    """
    task_label = f"Task"
    num_train = len(task.train)
    num_test = len(task.test)
    total_examples = num_train + num_test

    if total_examples == 0:
        print(f"{task_label} has no examples to visualize.")
        return

    # Create a figure with 2 rows (input/output) and columns for each example
    # Adjust figsize dynamically: make width proportional to number of examples
    fig_width = max(5, 1.5 * total_examples) # Further reduced width scaling
    fig_height = 5 # Further reduced height
    fig, axes = plt.subplots(2, total_examples, figsize=(fig_width, fig_height), squeeze=False)

    # --- Plot Training Examples --- 
    for i, example in enumerate(task.train):
        visualize_grid(example.input, ax=axes[0, i], title=f"Train {i} In", show_gridlines=True, show_attributes_in_title=False)
        visualize_grid(example.output, ax=axes[1, i], title=f"Train {i} Out", show_gridlines=True, show_attributes_in_title=False)
    
    # --- Plot Test Examples --- 
    for i, example in enumerate(task.test):
        col_idx = num_train + i
        visualize_grid(example.input, ax=axes[0, col_idx], title=f"Test {i} In", show_gridlines=True, show_attributes_in_title=False)
        # Assuming test examples *do* have an output grid for visualization purposes
        visualize_grid(example.output, ax=axes[1, col_idx], title=f"Test {i} Out", show_gridlines=True, show_attributes_in_title=False)

    # Construct the super title including task attributes (using correct names)
    sup_title = f"{task_label}"
    parts = extract_attributes(task)
    attribute_text = " | ".join(parts)
    fig.suptitle(f"{sup_title}\n{attribute_text}", fontsize=10) # Reduced font size further
    # Adjust layout to prevent overlap and accommodate title
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Adjust top margin more
    plt.show()


if __name__ == "__main__":
    task_idx = 1
    task = ARCAGI1_EVAL.tasks[task_idx]
    example = task.train[0]
    grid = example.input

    print("\n--- Visualizing Single Grid (with Attributes) ---")
    visualize_grid(grid) 
    
    print("\n--- Visualizing Example (with Attributes) ---")
    visualize_example(example)
    
    print("\n--- Visualizing Task (Task Attributes Only) ---")
    visualize_task(task)
