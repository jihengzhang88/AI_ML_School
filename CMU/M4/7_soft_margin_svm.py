import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


def plot_boundary(x, y, w1, w2, b, e=0.1):
    x1min, x1max = min(x[:, 0]), max(x[:, 0])
    x2min, x2max = min(x[:, 1]), max(x[:, 1])

    xb = np.linspace(x1min, x1max)
    y_0 = 1 / w2 * (-b - w1 * xb)
    y_1 = 1 / w2 * (1 - b - w1 * xb)
    y_m1 = 1 / w2 * (-1 - b - w1 * xb)

    cmap = ListedColormap(["purple", "orange"])

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap)
    plt.plot(xb, y_0, '-', c='blue')
    plt.plot(xb, y_1, '--', c='green')
    plt.plot(xb, y_m1, '--', c='green')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis((x1min - e, x1max + e, x2min - e, x2max + e))


data = np.load("w4-hw1-data.npy")
X = data[:, 0:2]
y = data[:, 2]


def soft_margin_svm(X, y, C):
    N = np.shape(X)[0]

    # YOUR CODE GOES HERE
    # Define P, q, G, h
    P_ul = np.eye(3)
    P_ul[-1, -1] = 0
    P = np.block([[P_ul, np.zeros((3, N))], [np.zeros((N, 3)), np.zeros((N, N))]])

    q_top = np.zeros(3)
    q_bottom = C * np.ones(N)
    q = np.concatenate([q_top, q_bottom])

    G_ul = -y.reshape(-1, 1) * np.hstack([X, np.ones((N, 1))])
    G_ur = -np.eye(N)
    G_ll = np.zeros((N, 3))
    G_lr = -np.eye(N)
    G = np.block([[G_ul, G_ur], [G_ll, G_lr]])

    h_top = -np.ones(N)
    h_bottom = np.zeros(N)
    h = np.concatenate([h_top, h_bottom])

    z = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    w1 = z['x'][0]
    w2 = z['x'][1]
    b = z['x'][2]

    return w1, w2, b


# YOUR CODE GOES HERE
for C in [1e-5, 1e-3, 1e-2, 0.05, 1]:
    w1, w2, b = soft_margin_svm(X, y, C)
    print(f"\nSolution\n--------\nw1: {w1:8.4f}\nw2: {w2:8.4f}\n b: {b:8.4f}")

    plt.figure()
    plot_boundary(X,y,w1,w2,b,e=1)
    plt.title(f"C = {C}")
    plt.show()
