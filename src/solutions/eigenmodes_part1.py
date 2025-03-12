import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def create_init_matrix_a(N):
    """
    Create an initial dependency matrix for an N x N grid.

    Parameters:
        N (int): The number of rows (and columns) in the grid. The total number of points in the
        grid is N*N.

    Returns
    -------
    numpy.ndarray
        A two-dimensional NumPy array of shape (N*N, N*N) representing the dependency
        matrix. Each row corresponds to a grid point, with the diagonal containing the
        adjusted dependency value and the neighboring positions (if applicable) set to 1.
    """
    # initial matrix has (N*N)*(N*N) size, to capture all the dependencies
    initial_matrix = np.zeros((N * N, N * N))
    for i, row in enumerate(initial_matrix):
        # booleans for skipping rows
        skip_first_row = False
        skip_first_col = False
        skip_last_col = False
        skip_last_row = False

        # if no neighbor point of border point or border point the diagonal value is 4
        waarde = 4

        # adress border points or neighbors of border points
        if i % N == 0:
            continue
        if i % N == N - 1:
            continue
        if i % N <= 1:
            waarde -= 1
            skip_first_col = True
        if i % N >= N - 2:
            waarde -= 1
            skip_last_col = True
        if i < N:
            waarde -= 1
            skip_first_row = True
        if i >= N * (N - 1):
            waarde -= 1
            skip_last_row = True

        # assign values for the dependencies
        row[i] = -waarde
        if not skip_last_col:
            row[i + 1] = 1
        if not skip_first_col:
            row[i - 1] = 1
        if not skip_last_row:
            row[i + N] = 1
        if not skip_first_row:
            row[i - N] = 1
    return initial_matrix


def create_circle(N):
    # define midpoint (correct for zero indexing)
    mid = 0.5 * (N - 1)

    # initialize circle grid
    initial_circle = np.zeros((N, N))
    for i, row in enumerate(initial_circle):
        for j, _ in enumerate(row):
            # calculate euclidean distance to midpoint
            if np.sqrt((i - mid) ** 2 + (j - mid) ** 2) <= 0.5 * (N):
                # if cell lies within radius from midpoint, set index to 1
                initial_circle[i, j] = 1

    # set border to -1
    for i, row in enumerate(initial_circle):
        for j, _ in enumerate(row):
            boudnary = False
            if initial_circle[i, j] == 0:
                continue
            if i == 0 or j == 0 or i == N - 1 or j == N - 1:
                boudnary = True

            # check for neighboring values
            top = i > 0 and initial_circle[i - 1, j] == 0
            bottom = i < N - 1 and initial_circle[i + 1, j] == 0
            left = j > 0 and initial_circle[i, j - 1] == 0
            right = j < N - 1 and initial_circle[i, j + 1] == 0
            if top or bottom or left or right:
                boudnary = True

            if boudnary:
                initial_circle[i, j] = -1
    return initial_circle


def create_circle_dependency(N, initial_circle):
    """
    Create an initial dependency matrix for an N x N grid.

    Parameters:
        N (int): The number of rows (and columns) in the grid. The total number of points in the
        grid is N*N.

    Returns
    -------
    numpy.ndarray
        A two-dimensional NumPy array of shape (N*N, N*N) representing the dependency
        matrix. Each row corresponds to a grid point, with the diagonal containing the
        adjusted dependency value and the neighboring positions (if applicable) set to 1.
    """
    # initial matrix has (N*N)*(N*N) size, to capture all the dependencies
    initial_matrix = np.zeros((N * N, N * N))
    for i, row in enumerate(initial_matrix):
        row_index = int(i / N)
        col_index = i % N

        if initial_circle[row_index, col_index] == 0:
            # outside circle area
            row[i] = 1
            continue
        if initial_circle[row_index, col_index] == -1:
            row[i] = 1
            continue

        # booleans for skipping rows
        skip_first_row = False
        skip_first_col = False
        skip_last_col = False
        skip_last_row = False

        # if no neighbor point of border point or border point the diagonal value is 4
        waarde = 4

        # adress border points or neighbors of border points

        if initial_circle[row_index, col_index - 1] == -1:
            waarde -= 1
            # skip_first_col = True
        if initial_circle[row_index, col_index + 1] == -1:
            waarde -= 1
            # skip_last_col = True
        if initial_circle[row_index - 1, col_index] == -1:
            waarde -= 1
            # skip_first_row = True
        if initial_circle[row_index + 1, col_index] == -1:
            waarde -= 1
            # skip_last_row = True

        # assign values for the dependencies
        row[i] = -waarde
        # if not skip_last_col:
        #     row[i + 1] = 1
        # if not skip_first_col:
        #     row[i - 1] = 1
        # if not skip_last_row:
        #     row[i + N] = 1
        # if not skip_first_row:
        #     row[i - N] = 1

        row[i + 1] = 1
        row[i - 1] = 1
        row[i + N] = 1
        row[i - N] = 1
    return initial_matrix


N = 30
matrix = create_init_matrix_a(N)
# eigenvalues, eigenvectors = eigsh(matrix, k=3, which="SM")
eigenvalues, eigenvectors = eigh(matrix)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors: {eigenvectors}")

sorted_indices = np.argsort(eigenvalues)
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

num_vectors_to_plot = 3
selected_eigenvectors = sorted_eigenvectors[:, :num_vectors_to_plot]

grid_size = (N, N)
reshaped_full = [vec.reshape(grid_size) for vec in selected_eigenvectors.T]
reshaped_eigenvectors = [grid[1:-1, 1:-1] for grid in reshaped_full]

N = 70
# create circle grid
circle = create_circle(N)
matrix_circle = create_circle_dependency(N, circle)
sparse_matrix_circle = csr_matrix(matrix_circle)
eigenvalues_circle, eigenvectors_circle = eigsh(sparse_matrix_circle, k=3, which="SM")

sorted_indices_circle = np.argsort(eigenvalues_circle)
sorted_eigenvalues_circle = eigenvalues_circle[sorted_indices_circle]
sorted_eigenvectors_circle = eigenvectors_circle[:, sorted_indices_circle]

num_vectors_to_plot = 3
selected_eigenvectors_circle = sorted_eigenvectors_circle[:, :num_vectors_to_plot]

grid_size = (N, N)
reshaped_full_circle = [
    vec.reshape(grid_size) for vec in selected_eigenvectors_circle.T
]
reshaped_eigenvectors_circle = [grid[1:-1, 1:-1] for grid in reshaped_full_circle]

all_vals_circ = np.concatenate([arr.flatten() for arr in reshaped_eigenvectors_circle])
vmin_circ, vmax_circ = all_vals_circ.min(), all_vals_circ.max()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes[0, :]):
    im = ax.imshow(reshaped_eigenvectors[i], cmap="coolwarm")
    ax.set_title(f"Eigenvalue: {sorted_eigenvalues[i]:.2f}")
    plt.colorbar(im, ax=ax)

for i, ax in enumerate(axes[1, :]):
    im_circ = ax.imshow(
        reshaped_eigenvectors_circle[i], cmap="coolwarm", vmin=vmin_circ, vmax=vmax_circ
    )
    ax.set_title(f"Eigenvalue: {sorted_eigenvalues_circle[i]:.2f}")

cbar_circ = fig.colorbar(im_circ, ax=axes[1, :], orientation="vertical")
cbar_circ.set_label("Circle amplitude")

plt.show()
