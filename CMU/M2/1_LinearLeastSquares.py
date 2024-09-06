import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.array([[-1, 3, 5]]).T
y = np.array([[2, -1, 4]]).T

fig, ax = plt.subplots()
plt.plot(x, y,'o')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
sns.despine()
plt.grid()
plt.show()


bias = np.ones_like(x)

X = np.concatenate([x,bias],1)

print("Design Matrix:\n",X)


# YOUR CODE GOES HERE
# Get coefficients w
w = np.linalg.inv(X.T @ X) @ X.T @ y
w = w.flatten()
print("Linear Coefficients:\n", w)


n = 40
x_test = np.linspace(-4,7,n).reshape(-1,1)
bias_test = np.ones_like(x_test)
X_test = np.concatenate([x_test, bias_test], 1)

# YOUR CODE GOES HERE
w = np.linalg.inv(X.T @ X) @X.T @y
# Predict y_test
y_test = X_test @ w

fig, ax = plt.subplots()
plt.plot(x, y,'.')
plt.plot(x_test, y_test)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
sns.despine()
plt.grid()
plt.show()