import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2 + 3*x + 6*np.sin(x)


def plotfx():
    # Sample function
    xs = np.linspace(-12,10,100)
    ys = f(xs)
    # Plot function
    plt.plot(xs,ys,'r-')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.show()


# Visualize the function
plotfx()


# Your fgrad(x) function goes here
def fgrad(x):
    return 2*x + 3 + 6*np.cos(x)


iter = 10
eta = 0.15
x = 8

for i in range(iter):
    # YOUR GRADIENT DESCENT CODE GOES HERE
    x -= eta*fgrad(x)
    print('Iteration %d, x = %.3f, f(x) = %.3f' %(i+1, x, f(x)))