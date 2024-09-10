import numpy as np
import matplotlib.pyplot as plt


def gt_function():
    xt = np.linspace(0,1,101)
    yt = np.sin(2 *np.pi*xt)
    return xt, yt


def plot_data(x,y,xt,yt,title = None):
    # Provide title as a string e.g. 'string'
    plt.plot(x,y,'bo',label = 'Data')
    plt.plot(xt,yt,'g-', label = 'Ground Truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def plot_model(x,y,xt,yt,xr,yr,title = None):
    # Provide title as a string e.g. 'string'
    plt.plot(x,y,'bo',label = 'Data')
    plt.plot(xt,yt,'g-', label = 'Ground Truth')
    plt.plot(xr,yr,'r-', label = 'Fitted Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


# YOUR CODE GOES HERE
xt, yt = gt_function()
x10 = np.load('d10.npy')[:, 0]
y10 = np.load('d10.npy')[:, 1]
x100 = np.load('d100.npy')[:, 0]
y100 = np.load('d100.npy')[:, 1]
plot_data(x10,y10,xt,yt,title = '10 data')
plot_data(x100,y100,xt,yt,title = '100 data')


# YOUR CODE GOES HERE
def makeModel(x, y, m):
    X = np.vander(x, m+1)
    w = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1, 1)
    xr = np.linspace(0, 1, 101)
    Xr = np.vander(xr, m+1)
    yr = Xr @ w
    print(f"w = {w.flatten()}")
    plot_model(x, y, xt, yt, xr, yr, title = f"{m}nd order polynomial fit to d{x.size}")


makeModel(x10, y10, 2)
makeModel(x100, y100, 2)
makeModel(x10, y10, 9)
makeModel(x100, y100, 9)


# YOUR CODE GOES HERE
def makeL2Model(x, y, m):
    X = np.vander(x, m+1)
    I_m = np.eye(m+1)
    I_m[-1, -1] = 0
    w = np.linalg.inv(X.T @ X + np.exp(-10) * I_m) @ X.T @ y.reshape(-1, 1)
    xr = np.linspace(0, 1, 101)
    Xr = np.vander(xr, m+1)
    yr = Xr @ w
    print(f"w = {w.flatten()}")
    plot_model(x, y, xt, yt, xr, yr, title = f"{m}th order polynomial fit to d{x.size}")


makeL2Model(x10, y10, 9)
makeL2Model(x100, y100, 9)


# YOUR CODE GOES HERE
def grad(w, X, y, I_m):
    return X.T @ (X @ w - y10.reshape(-1, 1)) + np.exp(-10) * I_m @ w


def makeL2GDModel(x, y, m):
    X = np.vander(x, m + 1)
    I_m = np.eye(m + 1)
    I_m[-1, -1] = 0
    w = np.zeros((m + 1, 1))
    for i in range(50000):
        w = w - 0.075 * grad(w, X, y, I_m)

    xr = np.linspace(0, 1, 101)
    Xr = np.vander(xr, m + 1)
    yr = Xr @ w

    print(f"w = {w.flatten()}")
    plot_model(x, y, xt, yt, xr, yr, title=f"{m}th order polynomial fit to d{x.size}")


makeL2GDModel(x10, y10, 9)