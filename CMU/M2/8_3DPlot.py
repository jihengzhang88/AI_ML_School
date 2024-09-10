import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# YORU CODE GOES HERE
x = np.load('tempfield.npy')[:, 0]
y = np.load('tempfield.npy')[:, 1]
z = np.load('tempfield.npy')[:, 2]
T = np.load('tempfield.npy')[:, 3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=T, cmap='jet')

cbar = plt.colorbar(sc, ax=ax, label='T')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with T')
plt.show()

# YOUR CODE GOES HERE
X = np.vstack([x, y, z, np.ones_like(x)]).T
w = np.linalg.inv(X.T @ X) @ X.T @ T.reshape(-1, 1)
T_pred = np.array([5, 5, 5, 1]) @ w
print('T(5, 5, 5) = %.5f Celsius' % T_pred.item())

gradient = w[:3].flatten()
direction_of_decrese = -gradient
direction_of_decrese_normalized = direction_of_decrese / np.linalg.norm(direction_of_decrese)
print("Direction of maximum decrease in temperaure:", direction_of_decrese_normalized)