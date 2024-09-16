import numpy as np
import matplotlib.pyplot as plt


def sigmoid(h):
    # YOUR CODE GOES HERE
    return 1/(1 + np.exp(-h))


def transform_quadratic(x, w):
    # YOUR CODE GOES HERE
    return np.array([1, x, x**2]) @ w


w = [4,-3,2]
for x in [1.2, 7.]:
    P = sigmoid(transform_quadratic(x,w))
    print(f"x = {x:3} --> P(y=1) = {P}")