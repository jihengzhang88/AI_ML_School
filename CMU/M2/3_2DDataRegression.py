# Generating data for the problem
import numpy as np
import matplotlib.pyplot as plt


def gaussian2d(A, mx, my, sx, sy):
    F = lambda xy: A*np.exp(-((xy[:,0]-mx)**2/(2*sx*sx)
                            + (xy[:,1]-my)**2/(2*sy*sy)))
    return F


def get_data_function():
    f1 = gaussian2d(A=0.7, mx = 0.25,my=0.25,sx=0.25,sy=0.25)
    f2 = gaussian2d(A=0.7, mx = 0.75,my=0.75,sx=0.25,sy=0.45)
    f = lambda xy: f1(xy) + f2(xy)
    return f


np.random.seed(0)
x = np.random.rand(60,2)
f = get_data_function()
y = f(x)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Writing a 3D Plotting function. Inputs data points and regression function
def plot_data_with_regression(x_data, y_data, regfun=None):
    plt.figure(figsize=(8,8))
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data[:,0], x_data[:,1],0*y_data,s=13,c=y_data,zorder=-1,cmap="coolwarm", alpha=1, edgecolor="black", linewidth=0.2)
    ax.scatter(x_data[:,0], x_data[:,1],y_data,s=20,c="black",zorder=-1)
    for i in range(len(y_data)):
        ax.plot([x_data[i,0],x_data[i,0]],[x_data[i,1],x_data[i,1]],[0,y_data[i]],'k:',linewidth=0.3)

    ax.set_xlabel('\n' + r"$x_1$")
    ax.set_ylabel('\n' + r"$x_2$")
    ax.set_zlabel('\n'+r"$y$")
    ax.set_zlim(0,0.9)

    if regfun is not None:
        vals = np.linspace(0, 1, 100)
        x1grid, x2grid = np.meshgrid(vals, vals)
        y = regfun(np.concatenate((x1grid.reshape(-1,1),x2grid.reshape(-1,1)),1)).reshape(*np.shape(x1grid))
        ax.plot_surface(x1grid, x2grid, y.reshape(x1grid.shape), alpha = 0.8, cmap = cm.coolwarm)
        plt.show()


plot_data_with_regression(x,y)


def get_linear_design_matrix(x):
    x1 = x[:,0].reshape(-1, 1)
    x2 = x[:,1].reshape(-1, 1)
    columns = [x1, x2, np.ones_like(x1)]   # Linear design matrix has a column of x1, column of x2, and a column of ones
    X = np.concatenate(columns, axis=1)   # Combine each column horizontally to make a matrix
    return X


X = get_linear_design_matrix(x)
print("First four rows of X:")
print(X[:4,:])


# Get coefficients
w1 = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1,1)
print("Linear Coefficients:", w1.flatten())


def do_2d_linear_regression(x):
    y_fit = get_linear_design_matrix(x) @ w1
    return y_fit

plot_data_with_regression(x, y, do_2d_linear_regression)


def get_quadratic_design_matrix(x):
    x1 = x[:,0].reshape(-1, 1)
    x2 = x[:,1].reshape(-1, 1)

    # YOUR CODE GOES HERE
    columns = [x1**2, x2**2, x1*x2, x1, x2, np.ones_like(x1)]
    X = np.concatenate(columns, axis=1)
    # 2ND ORDER, 2-D DESIGN MATRIX NEEDS 6 TOTAL COLUMNS

    return X


X = get_quadratic_design_matrix(x)
print("First four rows of X:")
print(X[:4,:])


# Get coefficients
w2 = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1,1)
print("Quadratic Coefficients:", w2.flatten())


def do_2d_quadratic_regression(x):
    y_fit = get_quadratic_design_matrix(x) @ w2
    return y_fit

plot_data_with_regression(x, y, do_2d_quadratic_regression)