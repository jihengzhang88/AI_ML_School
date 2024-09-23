import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


def plot_data(x, y, e=0.1):
    x1min, x1max = min(x[:, 0]), max(x[:, 0])
    x2min, x2max = min(x[:, 1]), max(x[:, 1])

    xb = np.linspace(x1min, x1max)

    cmap = ListedColormap(["blue", "red"])

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap)
    plt.colorbar()

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis((x1min - e, x1max + e, x2min - e, x2max + e))


def plot_SV_decision_boundary(svm, extend=True):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]

    x = np.linspace(xlim[0] - extend * xrange, xlim[1] + extend * xrange, 100)
    y = np.linspace(ylim[0] - extend * yrange, ylim[1] + extend * yrange, 100)

    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = svm.decision_function(xy)

    P = P.reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[0], linestyles=['-'])
    ax.contour(X, Y, P, colors='k', levels=[-1, 1], alpha=0.6, linestyles=['--'])

    plt.xlim(xlim)
    plt.ylim(ylim)


relative_compactness = np.array([0.98, 0.9, 0.86, 0.82, 0.79, 0.76, 0.74, 0.71, 0.69, 0.66, 0.64,
                                 0.62])
wall_area = np.array([294., 318.5, 294., 318.5, 343., 416.5, 245., 269.5, 294.,
                      318.5, 343., 367.5])
heating_load = np.array([24.58, 29.03, 26.28, 23.53, 35.56, 32.96, 10.36, 10.71, 11.11,
                         11.68, 15.41, 12.96])

X = np.vstack([relative_compactness, wall_area]).T
heating_class = np.where(heating_load < 20, -1, 1)
clf = SVC(C=1e5, kernel='linear')
clf.fit(X, heating_class)
y = clf.predict(X)
plt.figure(figsize=(6,4),dpi=150)
plot_data(X,y)
plot_SV_decision_boundary(clf, extend=True)
plt.xlabel("Relative Compactness")
plt.ylabel("Wall Area")
plt.title("Heating Load High/Low")
plt.show()
