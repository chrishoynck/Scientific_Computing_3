import numpy as np


def create_init_matrix_a(N, rectangular=False):
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
    if rectangular:
        initial_matrix = np.zeros((2 * N * N, 2 * N * N))
    else:
        initial_matrix = np.zeros((N * N, N * N))

    for i, row in enumerate(initial_matrix):
        # booleans for skipping rows

        row_index = int(i / N)
        col_index = i % N

        if rectangular:
            row_index = i // (2 * N)
            col_index = i % (2 * N)

        # if no neighbor point of border point or border point the diagonal value is 4
        waarde = 4

        # adress border points or neighbors of border points
        if rectangular:
            if (
                row_index == 0
                or col_index == 0
                or row_index == N - 1
                or col_index == 2 * N - 1
            ):
                row[i] = 1
                continue
            row[i] = -waarde

            row[i + 1] = 1
            row[i - 1] = 1
            row[i + 2 * N] = 1
            row[i - 2 * N] = 1
        else:
            if (
                row_index == 0
                or col_index == 0
                or row_index == N - 1
                or col_index == N - 1
            ):
                row[i] = 1
                continue

            row[i] = -waarde

            row[i + 1] = 1
            row[i - 1] = 1
            row[i + N] = 1
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

        # if no neighbor point of border point or border point the diagonal value is 4
        waarde = 4
        # assign values for the dependencies
        row[i] = -waarde

        row[i + 1] = 1
        row[i - 1] = 1
        row[i + N] = 1
        row[i - N] = 1
    return initial_matrix
