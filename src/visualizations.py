import numpy as np
import matplotlib.pyplot as plt
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
    X, Y = np.meshgrid(range(matrix.shape[1] + 1),
                       range(matrix.shape[0] + 1))

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
    plt.figure(figsize=(3, 4))
    for ktj,(all_xs, all_vs) in data_per_k.items():
        plt.plot(all_xs, all_vs, label=f"k: {ktj}")

    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Harmonic Oscillator")
    plt.legend()
    plt.savefig("plots/harmonic_oscillator.png")
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
    for i, freq in enumerate(freqs):
        for (all_xs, all_vs) in data_per_freq[freq]:
            axs[i].plot(all_xs, all_vs, color="b")
        
        if i%2 == 0:
            axs[i].set_ylabel("v")
        if i>1:
            axs[i].set_xlabel("x")
        axs[i].set_title(frequencie_string + f"{freq}")
        # axs[i].set_legend()
    fig.suptitle("Phase Plot Oscillator with extra Force")
    plt.tight_layout()
    plt.savefig("plots/harmonic_oscillator_phase.png")
    plt.show()