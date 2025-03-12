import matplotlib.pyplot as plt
import numpy as np
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

print(f"Matrix M: {M}")
eigenvalues, eigenvectors = eigsh(M, k=5, which="SM")

print(f"Eigenvalues (related to the frequency of the modes): {eigenvalues}")
print(f"Eigenvectors (related to the shape of the modes): {eigenvectors}")

sorted_indices = np.argsort(eigenvalues)
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

num_vectors_for_plotting = 3
selected_eigenvectors = sorted_eigenvectors[:, :num_vectors_for_plotting]

origin = np.zeros((selected_eigenvectors.shape[1],))

plt.figure(figsize=(6, 6))

# Plot the first three modes
for i in range(num_vectors_for_plotting):
    plt.quiver(
        *origin,
        selected_eigenvectors[0, i],
        selected_eigenvectors[1, i],
        scale=5,
        scale_units="xy",
        angles="xy",
        label=f"Eigenvector {i + 1}",
    )

# plt.xlim(-0.25, 0.25)
# plt.ylim(-0.25, 0.25)
plt.axhline(0, color="grey", lw=0.5)
plt.axvline(0, color="grey", lw=0.5)
plt.legend()
plt.grid(True)
plt.title("Eigenvectors Corresponding to Smallest Eigenvalues")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
