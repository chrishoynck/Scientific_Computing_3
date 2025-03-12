from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

N = 4
L = 1.0
h = L / (N - 1)  # spacing between points

map_of_indices = {}
count = 0
for i in range(N):
    for j in range(N):
        # Assign unique value to each interior point (implicitly excluding boundary points = 0)
        if 0 < i < N - 1 and 0 < j < N - 1:
            map_of_indices[(i, j)] = count
            count += 1

M = lil_matrix((count, count))

for (i, j), index in map_of_indices.items():
    M[index, index] = -4  # For each interior point, assign the value -4 to the diagonal
    for di, dj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
        # Check if the neighbour is an interior point as well and if so, assign value of 1
        if (di, dj) in map_of_indices:
            neighbour_index = map_of_indices[(di, dj)]
            M[index, neighbour_index] = 1

M = M * (1 / h**2)
eigenvalues, eigenvectors = eigsh(M, k=3, which="SM")

print(f"Eigenvalues (related to the frequency of the modes): {eigenvalues}")
print(f"Eigenvectors (related to the shape of the modes): {eigenvectors}")
