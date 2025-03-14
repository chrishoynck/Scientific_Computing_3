import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def visualize_mesh(matrix):
    """
    Visualize a 2D integer matrix as a colored mesh.

    Each unique integer value in 'matrix' will be assigned its own color.
    A grid is drawn to highlight cell boundaries.

    Parameters
    ----------
    matrix : np.ndarray
        2D NumPy array containing integer values (e.g., -1, 0, 1).

    Returns
    -------
    None
        Displays the mesh plot with a colorbar.
    """
    # Find the unique values and sort them
    unique_vals = np.unique(matrix)

    # Define boundaries between the discrete values
    # For example, between -1 and 0, we put a boundary at -0.5, etc.
    boundaries = [val - 0.5 for val in unique_vals] + [unique_vals[-1] + 0.5]

    # Create a color map with as many distinct colors as there are unique values
    # You can customize the colormap if desired
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
    cmap = ListedColormap(colors)

    # Use BoundaryNorm so each integer value maps to its own color band
    norm = BoundaryNorm(boundaries, len(unique_vals))

    # Set up the figure
    plt.figure(figsize=(15, 10))

    # Create a mesh grid that goes one beyond the matrix shape
    # so each cell is drawn correctly by pcolormesh
    X, Y = np.meshgrid(range(matrix.shape[1] + 1), range(matrix.shape[0] + 1))

    # Plot the data as a colored mesh with grid lines
    plt.pcolormesh(X, Y, matrix, cmap=cmap, norm=norm)

    # Add a colorbar; set its ticks to the actual integer values
    cbar = plt.colorbar(ticks=unique_vals)
    cbar.set_label("Matrix Values")

    # Invert the y-axis so row 0 is at the top
    plt.gca().invert_yaxis()

    # Axis labels and title
    plt.title("Matrix Mesh Visualization")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")

    # Show the plot
    plt.show()

def visualize_different_shapes(
    eigenvectors_square,
    eigenvalues_square,
    eigenvectors_circle,
    eigenvalues_circle,
    eigenvectors_rect,
    eigenvalues_rect,
    grid_size,
    num_modes=3,
):
    # Process square eigenvectors
    sorted_idx_sq = np.argsort(eigenvalues_square)
    sorted_eigvals_sq = eigenvalues_square[sorted_idx_sq]
    sorted_eigvecs_sq = eigenvectors_square[:, sorted_idx_sq]
    reshaped_sq = [vec.reshape((grid_size, grid_size)) for vec in sorted_eigvecs_sq.T[:num_modes]]

    # Process circle eigenvectors
    sorted_idx_circ = np.argsort(eigenvalues_circle)
    sorted_eigvals_circ = eigenvalues_circle[sorted_idx_circ]
    sorted_eigvecs_circ = eigenvectors_circle[:, sorted_idx_circ]
    reshaped_circ = [vec.reshape((grid_size, grid_size)) for vec in sorted_eigvecs_circ.T[:num_modes]]

    # Process rectangle eigenvectors
    sorted_idx_rect = np.argsort(eigenvalues_rect)
    sorted_eigvals_rect = eigenvalues_rect[sorted_idx_rect]
    sorted_eigvecs_rect = eigenvectors_rect[:, sorted_idx_rect]
    reshaped_rect = [vec.reshape((grid_size, 2 * grid_size)) for vec in sorted_eigvecs_rect.T[:num_modes]]

    # Compute global vmin and vmax for color scaling
    all_values = reshaped_sq + reshaped_circ + reshaped_rect
    global_vmin = min(arr.min() for arr in all_values)
    global_vmax = max(arr.max() for arr in all_values)

    # Figure + colorbar
    fig, axes = plt.subplots(3, num_modes, figsize=(5, 6))
    fig.subplots_adjust(bottom=0.14, left=0.2, wspace=0.1, hspace=0.25)

    # Add row labels for shape names
    row_labels = ["Square", "Circle", "Rectangle"]
    for i, label in enumerate(row_labels):
        axes[i, 0].annotate(
            label, xy=(-0.6, 0.5), xycoords="axes fraction",
            fontsize=12, ha='right', va='center', rotation=90
        ) 

    # Plot Square
    for i in range(num_modes):
        im = axes[0, i].imshow(reshaped_sq[i], cmap="viridis", vmin=global_vmin, vmax=global_vmax, origin="lower")
        axes[0, i].set_title(f"ω = {sorted_eigvals_sq[i]:.4f}", fontsize=10)

    # Plot Circle
    for i in range(num_modes):
        axes[1, i].imshow(reshaped_circ[i], cmap="viridis", vmin=global_vmin, vmax=global_vmax, origin="lower")
        axes[1, i].set_title(f"ω = {sorted_eigvals_circ[i]:.4f}", fontsize=10)

    # Plot Rectangle
    for i in range(num_modes):
        axes[2, i].imshow(reshaped_rect[i], cmap="viridis", vmin=global_vmin, vmax=global_vmax, origin="lower")
        axes[2, i].set_title(f"ω = {sorted_eigvals_rect[i]:.4f}", fontsize=10)

    # Tick label
    for row in range(3):
        for col in range(num_modes):
            if col > 0:  # Remove y-axis ticks from all but the first column
                axes[row, col].set_yticks([])
            if row == 0:  # Remove x-axis ticks from the first row
                axes[row, col].set_xticks([])

    # X-axis label
    for ax in axes[-1, :]:  
        ax.set_xlabel("x", fontsize=10, labelpad=1)

    # Y-axis label
    for ax in axes[:, 0]:  
        ax.set_ylabel("y", fontsize=10, labelpad=2)

    fig.suptitle("Eigenvectors for the 3 Smallest ω", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, location="bottom", shrink=1, aspect=30, pad=0.12)
    cbar.set_label("Magnitude", fontsize=10)

    plt.show()