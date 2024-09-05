import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.2855, 0.0033, 0.8307, 0.9606, 0.8153, 0.5539, 0.5152, 0.7761,
       0.5763, 0.2697, 0.6744, 0.7998, 0.1052, 0.8674, 0.598 , 0.3985,
       0.0171, 0.1732, 0.7976, 0.4137, 0.7161, 0.7225, 0.3892, 0.0834,
       0.9733, 0.3097, 0.8509, 0.0226, 0.6901, 0.2235, 0.5914, 0.5436,
       0.7189, 0.4558, 0.8366, 0.534 , 0.214 , 0.9314, 0.4065, 0.788 ])

y = np.array([ 0.6603, -0.5925,  0.2045,  0.3698,  0.191 ,  0.496 ,  0.4986,
        0.2516,  0.3905,  0.5932,  0.2924,  0.219 ,  0.0287,  0.2024,
        0.4489,  0.6237, -0.4857,  0.3384,  0.162 ,  0.6694,  0.2539,
        0.1936,  0.6322, -0.0953,  0.4632,  0.6721,  0.2464, -0.4672,
        0.2746,  0.5087,  0.3691,  0.4559,  0.2021,  0.5797,  0.2531,
        0.5417,  0.4577,  0.2952,  0.5856,  0.1818])


def plot_data_with_regression(x_data, y_data, x_reg, y_reg):
    plt.figure()

    plt.scatter(x_data, y_data, label="Data", c="black")
    plt.plot(x_reg, y_reg, label="Fit")

    plt.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$y$")
    plt.show()

# Function to get a linear design matrix for 1D data
def get_linear_design_matrix(x):
    x = x.reshape(-1, 1)                # Turn x into a column array
    columns = [x, np.ones_like(x)]      # Linear design matrix has a column of x and a column of ones
    X = np.concatenate(columns, axis=1) # Combine each column horizontally to make a matrix
    return X


X = get_linear_design_matrix(x)
print("First four rows of X:")
print(X[:4,:])

# Get coefficients
w = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1,1)
print("Linear Coefficients:", w.flatten())


x_fit = np.linspace(0, 1, 100)  # x values for fit line
y_fit = get_linear_design_matrix(x_fit) @ w  # y values for fit line

plot_data_with_regression(x, y, x_fit, y_fit)


def get_quadratic_design_matrix(x):
    # YOUR CODE GOES HERE
    # GENERATE A DESIGN MATRIX WITH 2ND ORDER FEATURES: X
    x = x.reshape(-1, 1)                # Turn x into a column array
    columns = [x**2, x, np.ones_like(x)]      # Linear design matrix has a column of x**2 and a colum of x and a column of one
    X = np.concatenate(columns, axis=1) # Combine each column horizontally to make a matrix
    return X


X = get_quadratic_design_matrix(x)
print("First four rows of X:")
print(X[:4,:])


# YOUR CODE GOES HERE
# COMPUTE COEFFICIENTS w
w = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1,1)
print("Quadratic Coefficients:", w.flatten())


# YOUR CODE GOES HERE
# PLOT
x_fit = np.linspace(0, 1, 100)                # x values for fit line
y_fit = get_quadratic_design_matrix(x_fit) @ w   # y values for fit line

plot_data_with_regression(x, y, x_fit, y_fit)


def get_cubic_design_matrix(x):
    # YOUR CODE GOES HERE
    # GENERATE A DESIGN MATRIX WITH 4TH ORDER FEATURES: X
    x = x.reshape(-1, 1)                # Turn x into a column array
    columns = [x**3, x**2, x, np.ones_like(x)]      # Linear design matrix has a column of x**3, a colum of x**2 and a colum of x and a column of one
    X = np.concatenate(columns, axis=1) # Combine each column horizontally to make a matrix
    return X


X = get_cubic_design_matrix(x)
print("First four rows of X:")
print(X[:4,:])


# YOUR CODE GOES HERE
# COMPUTE COEFFICIENTS w
w = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1,1)
print("Cubic Coefficients:", w.flatten())


# YOUR CODE GOES HERE
# PLOT
x_fit = np.linspace(0, 1, 100)                # x values for fit line
y_fit = get_cubic_design_matrix(x_fit) @ w   # y values for fit line

plot_data_with_regression(x, y, x_fit, y_fit)

