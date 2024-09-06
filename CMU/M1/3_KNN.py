import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 876
w1_data = np.random.uniform(-1,1,N)
w2_data = np.random.uniform(-1,1,N)
L_data = np.cos(4*w1_data + w2_data/4 - 1) + w2_data**2 + 2*w1_data**2

plt.figure(figsize=(5,4.2),dpi=120)
plt.scatter(w1_data,w2_data,s=10,c=L_data,cmap="jet")
plt.colorbar()
plt.axis("equal")
plt.xlabel("$w_1$")
plt.ylabel("$w_2$")
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.title("Data")
plt.show()


def distance(w1, w2):
    # YOUR CODE GOES HERE
    w1_diff = w1 - w1_data
    w2_diff = w2 - w2_data
    return np.sqrt(np.square(w1_diff) + np.square(w2_diff))
# Check that the function outputs the correct array size
assert(distance(0, 0).shape == w1_data.shape)


def get_knn_indices(w1, w2, k):
    d = distance(w1, w2)
    # YOUR CODE GOES HERE
    return np.argpartition(d, k)[:k]
# Check the function on the point w=(0,0) with k=5
indices = get_knn_indices(0,0,5)
print("5 points nearest (0,0):", indices, "\n(Should be 255 733 538 815 501)")


def knn_regress(w1, w2, k):
    indices = get_knn_indices(w1, w2, k)
    # YOUR CODE GOES HERE
    return np.mean(L_data[indices])
# Check the function on the point w=(0,0) with k=2
val = knn_regress(0,0,2)
print("Mean of 2 points nearest (0,0):", val, "\n(Should be about 0.72)")


w1_vals = np.linspace(-1,1,50)
w2_vals = np.linspace(-1,1,50)
print("w1 grid values:",w1_vals)
print("w2 grid values:",w2_vals)

w1s, w2s = np.meshgrid(w1_vals, w2_vals)
print("Size of w1 grid point array:", w1s.shape)
print("Size of w2 grid point array:", w2s.shape)

w1_grid, w2_grid = w1s.flatten(), w2s.flatten()
print("Flattened size of w1 grid point array:", w1_grid.shape)
print("Flattened size of w2 grid point array:", w2_grid.shape)

k = 20
L_grid = np.zeros_like(w1_grid)
for i in range(len(L_grid)):
    L_grid[i] = knn_regress(w1_grid[i], w2_grid[i],k)

plt.figure(figsize=(5,4.2),dpi=120)
plt.scatter(w1_grid,w2_grid,s=10,c=L_grid,cmap="jet")
plt.colorbar()
plt.axis("equal")
plt.xlabel("$w_1$")
plt.ylabel("$w_2$")
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.title("K-NN Regression")
plt.show()

