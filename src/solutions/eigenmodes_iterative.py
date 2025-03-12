import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


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


N = 50
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
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    im = ax.imshow(reshaped_eigenvectors[i], cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"Eigenmode {i + 1} (λ={eigenvalues[sorted_indices[i]]:.2f})")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
