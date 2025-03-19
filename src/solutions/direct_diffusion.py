import numpy as np
from scipy.sparse.linalg import spsolve
import src.solutions.eigenmodes_part1 as wave_sol
from scipy.sparse import csr_matrix

def direct_diffusion(nntjes, source_location, grid_size):
    """
    Computes the steady-state diffusion on a 2D circular domain for different grid discretizations.

    Parameters:
        nntjes (list of int): List of grid discretization sizes (N values) to be evaluated.
        source_location (tuple of float): Coordinates (x, y) of the diffusion source in the grid.
        grid_size (float): Physical size of the grid domain, ensuring all grids are scaled accordingly.

    Returns:
        list of tuples: Each tuple contains:
            - c_grid (numpy.ndarray): The computed steady-state diffusion grid for a given N.
            - outside_grid (numpy.ndarray): A mask indicating regions outside the circle (-1 values replaced by 1).
    """
    converged_grids = [] 
    xje, ytje = source_location

    # loop over all grid discretization sizes
    for N in nntjes:

        # create circle mask and dependency grid
        initial_circle = wave_sol.create_circle(N)
        dependency_circle = wave_sol.create_circle_dependency(N, initial_circle)

        # grid size should be positive, otherwise implementation doesn't work
        assert grid_size > 0, "grid size should be at least one cell"

        # determine flattened index with initial coordinates
        # which_to_initial = int((N)*N*ytje/(0.5*grid_size) + xje/(0.5*grid_size)*(N)) -N -1
        which_to_initial = int(((ytje + 0.5*grid_size)/(grid_size))*N*N + (xje + 0.5*grid_size)/grid_size *N) -N-1
        print(f"y: {which_to_initial//N}") 
        print(f"x: {which_to_initial%N}") 


        # intialize b with zeros, and set source index to 1
        b = np.zeros(N*N)
        b[which_to_initial] = 1

        # source is independent of every other cell, because it is fixed 
        dependency_circle[which_to_initial, :] = 0
        dependency_circle[which_to_initial, which_to_initial] = 1

        dependency_circle_sparse = csr_matrix(dependency_circle)

        # solve system, result is the diffusion grid
        x = spsolve(dependency_circle_sparse, b)
        c_grid = x.reshape((N, N))

        # mask grid cells outtside circle range
        outside_grid = np.copy(initial_circle)
        outside_grid[initial_circle==-1] = 1

        converged_grids.append((c_grid, outside_grid))

    return converged_grids


