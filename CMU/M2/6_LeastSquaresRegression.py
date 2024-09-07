import numpy as np
import matplotlib.pyplot as plt


def plot_data_with_regression(x_data, y_data, x_reg, y_reg, title=""):
    plt.figure()

    plt.scatter(x_data.flatten(), y_data.flatten(), label="Data", c="black")
    plt.plot(x_reg.flatten(), y_reg.flatten(), label="Fit")

    plt.legend(loc="upper left")
    plt.xlabel(r"$Re / 1000$")
    plt.ylabel(r"h, $W/m^2 K$")
    plt.xlim(0, 6)
    plt.ylim(50, 200)
    plt.title(title)
    plt.show()


deg = 5
x = np.array([1.010, 2.000, 2.990, 4.100, 5.020])
y = np.array([75.1, 104.0, 100.6, 138.8, 150.8])
X = np.vander(x, deg)

xreg = np.linspace(0, 6)
Xreg = np.vander(xreg, deg)

w = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1, 1)
yreg = Xreg @ w
plot_data_with_regression(x, y, xreg, yreg, "5th order polynomial regression")


def get_regularized_w(L):
    I_m = np.eye(deg)
    I_m[-1, -1] = 0
    # YOUR CODE GOES HERE
    w = np.linalg.inv(X.T @ X + L * I_m) @ X.T @ y.reshape(-1, 1)
    # return regularized w
    return w


for L in [0, .001, 0.1, 1000]:
    w = get_regularized_w(L)
    yreg = Xreg @ w
    plot_data_with_regression(x, y, xreg, yreg,f"Regression with Regularization, $\\lambda$={L}")
