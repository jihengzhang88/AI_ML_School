import numpy as np

def gini(D):
    # YOUR CODE GOES HERE
    sum_p = 0.0
    for i in D:
        pi = i/sum(D)
        sum_p += pi**2
    gini = 1- sum_p
    return gini


D = np.array([4, 9, 7, 0, 3])
g = gini(D)
print(f"gini([4,9,7,0,3]) = {g:.3f} (should be about {0.707})")

D1 = np.array([1,0,0])
D2 = np.array([0,0,4])
D3 = np.array([0, 20, 0, 0, 0, 3])
D4 = np.array([6, 6, 6, 6])

for D in [D1, D2, D3, D4]:
    # YOUR CODE GOES HERE
    g = gini(D)
    print(f"gini{D} = {g:.3f}")