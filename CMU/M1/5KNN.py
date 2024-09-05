import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


# Data generation -- don't change
np.random.seed(42)
N = 200
w1_data = np.random.uniform(-1,1,N)
w2_data = np.random.uniform(-1,1,N)
L_data = np.cos(4*w1_data) + np.sin(5*w2_data) + 2*w1_data**2 - w2_data/2
# (end of data generation)

plt.figure(figsize=(5,4.2),dpi=80)
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
    square_w1_diff = (w1 - w1_data) ** 2
    square_w2_diff = (w2 - w2_data) ** 2
    distance = np.sqrt(square_w1_diff + square_w2_diff)
    return distance

def get_knn_indices(d, k):
    return np.argpartition(d, k)[:k]

def weighted_knn(w1, w2, k):
    # YOUR CODE GOES HERE
    d = distance(w1, w2)
    indices = get_knn_indices(d, k)
    prox = 1/(d[indices] + 1e-9)
    percentage = prox/np.sum(prox)
    return np.dot(percentage, L_data[indices])

# YOUR CODE GOES HERE
# Visualize results for k = 1, 5, and 25
def plot(k):
    w1_vals = np.linspace(-1,1,50)
    w2_vals = np.linspace(-1,1,50)
    w1s, w2s = np.meshgrid(w1_vals, w2_vals)
    w1_grid, w2_grid = w1s.flatten(), w2s.flatten()
    L_grid = np.zeros_like(w1_grid)
    for i in range(len(L_grid)):
        L_grid[i] = weighted_knn(w1_grid[i], w2_grid[i],k)
    plt.figure(figsize=(5,4.2),dpi=120)
    plt.scatter(w1_grid,w2_grid,s=10,c=L_grid,cmap="jet")
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("$w_1$")
    plt.ylabel("$w_2$")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title(f"K-NN Weighted Regression with K = {k}")
    plt.show()

plot(1)
plot(5)
plot(25)


def create_fit(n_neighbors = 1):
    model = KNeighborsRegressor(n_neighbors, weights="distance")
    X = np.vstack([w1_data,w2_data]).T
    model.fit(X, L_data)
    return model


# YOUR CODE GOES HERE
# Visualize sklearn results for k = 1, 5, and 25
w1_vals = np.linspace(-1, 1, 50)
w2_vals = np.linspace(-1, 1, 50)
w1s, w2s = np.meshgrid(w1_vals, w2_vals)
w1_grid, w2_grid = w1s.flatten(), w2s.flatten()

for k in [1, 5, 25]:
    model = create_fit(k)
    L_grid = np.zeros_like(w1_grid)
    for i in range(len(L_grid)):
        L_grid[i] = model.predict(np.array([[w1_grid[i], w2_grid[i]]]))[0]

    plt.figure(figsize=(5, 4.2), dpi=120)
    plt.scatter(w1_grid, w2_grid, s=10, c=L_grid, cmap="jet")
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("$w_1$")
    plt.ylabel("$w_2$")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f"K-NN Weighted Regression with K = {k}")
    plt.show()