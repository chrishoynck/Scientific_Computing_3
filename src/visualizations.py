import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import src.solutions.eigenmodes_part1 as eigen_part1
import matplotlib.animation as animation
import matplotlib.ticker as ticker

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
    reshaped_sq = [
        vec.reshape((grid_size, grid_size)) for vec in sorted_eigvecs_sq.T[:num_modes]
    ]
    vmin_sq = min(arr.min() for arr in reshaped_sq)
    vmax_sq = max(arr.max() for arr in reshaped_sq)

    # Process circle eigenvectors
    sorted_idx_circ = np.argsort(eigenvalues_circle)
    sorted_eigvals_circ = eigenvalues_circle[sorted_idx_circ]
    sorted_eigvecs_circ = eigenvectors_circle[:, sorted_idx_circ]
    reshaped_circ = [
        vec.reshape((grid_size, grid_size)) for vec in sorted_eigvecs_circ.T[:num_modes]
    ]
    vmin_circ = min(arr.min() for arr in reshaped_circ)
    vmax_circ = max(arr.max() for arr in reshaped_circ)

    # Process rectangle eigenvectors
    sorted_idx_rect = np.argsort(eigenvalues_rect)
    sorted_eigvals_rect = eigenvalues_rect[sorted_idx_rect]
    sorted_eigvecs_rect = eigenvectors_rect[:, sorted_idx_rect]
    reshaped_rect = [
        vec.reshape((grid_size, 2 * grid_size))
        for vec in sorted_eigvecs_rect.T[:num_modes]
    ]
    vmin_rect = min(arr.min() for arr in reshaped_rect)
    vmax_rect = max(arr.max() for arr in reshaped_rect)

    fig, axes = plt.subplots(3, num_modes, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)

    # Square
    for i in range(num_modes):
        im_sq = axes[0, i].imshow(
            reshaped_sq[i], cmap="coolwarm", vmin=vmin_sq, vmax=vmax_sq
        )
        axes[0, i].set_title(f"Square: λ={sorted_eigvals_sq[i]:.2f}")
    fig.colorbar(im_sq, ax=axes[0, :], orientation="vertical")

    # Circle
    for i in range(num_modes):
        im_circ = axes[1, i].imshow(
            reshaped_circ[i], cmap="coolwarm", vmin=vmin_circ, vmax=vmax_circ
        )
        axes[1, i].set_title(f"Circle: λ={sorted_eigvals_circ[i]:.2f}")
    fig.colorbar(im_circ, ax=axes[1, :], orientation="vertical")

    # Rectangle
    for i in range(num_modes):
        im_rect = axes[2, i].imshow(
            reshaped_rect[i], cmap="coolwarm", vmin=vmin_rect, vmax=vmax_rect
        )
        axes[2, i].set_title(f"Rect: λ={sorted_eigvals_rect[i]:.2f}")
    fig.colorbar(im_rect, ax=axes[2, :], orientation="vertical")

    plt.show()

def eigenfrequencies_plot(sizes, eigenfrequencies_square, eigenfrequencies_circle, eigenfrequencies_rectangle):
    """
    Plots eigenfrequencies for Square, Circle, and Rectangle shapes for given sizes (L).
    
    Parameters:
        sizes (list): List of values for L (size parameter).
        eigenfrequencies_square (dict): Dictionary mapping L to eigenfrequencies for squares.
        eigenfrequencies_circle (dict): Dictionary mapping L to eigenfrequencies for circles.
        eigenfrequencies_rectangle (dict): Dictionary mapping L to eigenfrequencies for rectangles.
    """
    # Colours
    viridis = plt.cm.viridis
    colors = [viridis(0.2), viridis(0.5), viridis(0.8)]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)
    shapes = ["Square", "Circle", "Rectangle"]
    data_dicts = [eigenfrequencies_square, eigenfrequencies_circle, eigenfrequencies_rectangle]

    for ax, shape, color, data in zip(axes, shapes, colors, data_dicts):
        for N in sizes:
            if N in data:
                ax.scatter([N] * len(data[N]), data[N], color=color, s=1)
        ax.set_xlabel("L")
        ax.set_title(f"{shape}", fontsize=12, color="black")
        ax.grid(True)

    axes[0].set_ylabel("λ")
    plt.tight_layout()
    plt.show()

def plot_eigenmodes(N, num_modes, eigenvalues, eigenvectors, t_values, A, B, c):
    """
    Create an animation of eigenmodes over time.

    Parameters:
    - N (int): Matrix size.
    - num_modes (int): Number of eigenmodes.
    - eigenvalues (np.array): Computed eigenvalues.
    - eigenvectors (np.array): Computed eigenvectors.
    - t_values (np.array): Time steps for animation.
    - A, B, c (float): Constants for oscillation.
    """
    viridis = plt.cm.viridis
    colours = viridis(np.linspace(0, 1, num_modes))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    lines = [ax.plot([], [], marker='o', linestyle='-', color=colours[i])[0] for i in range(num_modes)]
    ax.set_xlim(0, N - 1)
    ax.set_ylim(-0.00075, 0.00075)
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax.set_xlabel("Grid Point")
    ax.set_ylabel("Amplitude")
    ax.set_title("Eigenmode Oscillations")
    fig.tight_layout()

    ani = animation.FuncAnimation(
        fig, eigen_part1.update, frames=len(t_values), init_func=lambda: eigen_part1.init(lines),
        fargs=(t_values, lines, eigenvalues, eigenvectors, A, B, c, ax, N), blit=False
    )

    # Animation
    #ani = animation.FuncAnimation(fig, eigen_part1.update, frames=len(t_values), init_func=eigen_part1.init(lines), blit=False)
    animation_filename = "plots/eigenmode_animation.gif"
    ani.save(animation_filename, writer="pillow", fps=20)

