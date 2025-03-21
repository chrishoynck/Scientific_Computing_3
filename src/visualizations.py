import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import src.solutions.eigenmodes_part1 as eigen_part1
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_mesh(matrix, ranges):
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

    plt.savefig("plots/eigenvectors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    colors = [viridis(0.1), viridis(0.5), viridis(0.9)]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(4, 1), sharey=True)
    shapes = ["Square", "Circle", "Rectangle"]
    data_dicts = [eigenfrequencies_square, eigenfrequencies_circle, eigenfrequencies_rectangle]

    for ax, shape, color, data in zip(axes, shapes, colors, data_dicts):
        for N in sizes:
            if N in data:
                ax.scatter([N] * len(data[N]), data[N], color=color, s=1)
        ax.set_xlabel("L")
        ax.set_title(f"{shape}", fontsize=12, color="black")
        ax.set_xticks([1, 2, 3, 4, 5])

    axes[0].set_ylabel("λ")

    plt.ylim(2.8, 3.2)
    plt.savefig("plots/eigenfrequencies.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    ani.save(animation_filename, writer="pillow", fps=10)

def plot_diffusion_circle(gridjes, Ntjes):
    """
    Visualizes the evolution of a 2D Diffusion on a circle

    Parameters:
        gridjes (Tuple): (grid, object_grid), where:
            - grid (numpy.ndarray): 2D array representing concentration values.
            - object_grid (numpy.ndarray): 2D array indicating circle placement.
        NNjtes (List(int)): list of different number of discretization steps used.
    """

    assert len(Ntjes) == 3, f"The number of different discretizations should be 3, now {len(Ntjes)}"

    # plot setup
    fig, axs = plt.subplots(1, 3, figsize=(4.9, 2.8), sharey=True)

    # colormaps
    object_cmap = mcolors.ListedColormap(["white", "none"])  # Only one color, yellow
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.viridis  # Choose a colormap
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)  

    deltax_string = r"$\Delta x: $"

    for i in range(3):
        gridd, object_gridd = gridjes[i]

        # fill in according to concentration value
        img = axs[i].imshow(
            gridd, cmap=cmap, norm=norm, origin="lower", extent=[-2, 2, -2, 2]
        )

        # mask cells lying outside circle
        axs[i].imshow(
            object_gridd, cmap=object_cmap, origin="lower", extent=[-2, 2, -2, 2] 
        )

        fraction_string = rf"$\frac{{1}}{{{Ntjes[i] // 4}}}$"

        axs[i].set_title(deltax_string +  fraction_string, fontsize=14)
        axs[i].set_xlabel("x")
    axs[0].set_ylabel("y")

    # colorbar settings
    cbar_ax = fig.add_axes([0.13, 0.07, 0.82, 0.03])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Concentration", fontsize=11)

    fig.suptitle("Circle Diffusion", fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.14, top=0.8, bottom=0.27)
    plt.savefig("plots/Diffusion_circle_a.png", dpi=300, bbox_inches="tight")
    plt.show()

def vis_harmonic_oscillator(data_per_k):
    """
    Visualizes the harmonic oscillator, computed with the leapfrog method

    Parameters:
        data_per_k (dict): k: (all_xs, all_vs), where:
            - k: spring constant (used for Hooke's law)
            - all_xs (list): List of computed spatial values
            - all_vs (list): List of computed velocities, corresponding with spatial value
    """
    plt.figure(figsize=(3.5, 4))

    ks = list(data_per_k.keys())
    colours = cm.viridis(np.linspace(0, 1, len(ks)))
    for (ktj,(all_xs, all_vs)), colour in zip(data_per_k.items(), colours):
        plt.plot(all_xs, all_vs, label=f"k: {ktj}", color=colour)

    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Harmonic Oscillator")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # legend placed outside of plot
    plt.savefig("plots/harmonic_oscillator.png", dpi=300, bbox_inches="tight")
    plt.show()

def vis_phase_oscillator(data_per_freq, freqs):
    """
    Visualizes the harmonic oscillator with extra sinusoidal force, computed with the leapfrog method

    Parameters:
        data_per_freq (dict): freq: (all_xs, all_vs), where:
            - freq: different frequencies used for extra sinusoidal force
            - all_xs (list): List of Lists of computed spatial values, one list for every initial x value
            - all_vs (list): List of Lists of computed velocity values, one list for every initial x value
    """

    frequencie_string = r"$\omega: $"
    fig, axs = plt.subplots(2, 2, figsize=(4, 4), sharey=False, sharex=False)
    axs = axs.flatten()

    colours = cm.viridis(np.linspace(0, 1, len(freqs)))

    for i, (freq, colour) in enumerate(zip(freqs, colours)):
        for (all_xs, all_vs) in data_per_freq[freq]:
            axs[i].plot(all_xs, all_vs, color=colour)
        
        if i%2 == 0:
            axs[i].set_ylabel("v")
        if i>1:
            axs[i].set_xlabel("x")
        axs[i].set_title(frequencie_string + f"{freq}")
        # axs[i].set_legend()
    fig.suptitle("Phase Plot Oscillator with Extra Force")
    plt.tight_layout()
    plt.savefig("plots/harmonic_oscillator_phase.png", dpi=300, bbox_inches="tight")
    plt.show()