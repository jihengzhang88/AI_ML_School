import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 50)
    y = np.linspace(ylim[0], ylim[1], 50)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1],
               linestyles=['--', '-', '--'],
               linewidths=[2, 4, 2])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


def plot_temp_profile(X, T, ax=None):
    if ax == None:
        ax = plt.gca()
    # Plot points colored by temperature
    sc = ax.scatter(X[:, 0], X[:, 1], c=T)
    # Add colorbar to plot
    cbar = plt.colorbar(sc)
    # Add labels
    cbar.set_label('Temperature ($\degree C$)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plot_temp_critical(X, y, ax=None):
    if ax is None:
        ax = plt.gca()
        showflag = True
    else:
        showflag = False
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['blue', 'red']))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect(0.8)
    if showflag:
        plt.show()
    else:
        return ax


def plot_model(model, X, y):
    # Wrapper function to generate plot and decision boundary
    ax = plt.gca()
    ax = plot_temp_critical(X, y, ax)
    plot_svc_decision_function(model, ax)


data = np.load('cputemp.npy')
X = data[:,:2]
T = data[:, 2]
plot_temp_profile(X,T)

y = np.where(T > 180, 1,0)
plot_temp_critical(X,y)


# Define accuracy function
def accuracy(model, X, y):
    preds = model.predict(X)
    accuracy = np.sum(preds == y) / len(y) *100
    return accuracy


# Train and plot SVC models
def model_train_plot(kernel, C, degree):
    model = SVC(kernel=kernel, degree=degree, C=C)
    model.fit(X, y)
    acc = accuracy(model, X, y)
    print(f'{kernel} model accuracy: {acc:.3f}')
    plot_model(model, X, y)
    return model


model_train_plot('rbf', 100, 8)
model_train_plot('poly', 100, 8)