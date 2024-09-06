import numpy as np
np.set_printoptions(precision=4)

y_data = np.array([[20,21,30,30,21,25,38,37,30,22,22,38,20,35]],dtype=np.double).T
y_pred = np.array([[21,21,31,30,20,28,36,32,31,20,21,39,21,34]],dtype=np.double).T

print("y_data = \n", y_data)
print("y_pred = \n", y_pred)

# YOUR CODE GOES HERE
# Compute y_err
y_err = y_data - y_pred
print("Size of y_err:", np.shape(y_err))

# YOUR CODE GOES HERE
# Compute MSE_loop
MSE_loop = 0.0
for err in y_err[:, 0]:
    MSE_loop += err*err
MSE_loop /= y_err.shape[0]
print("MSE (loop) = ", MSE_loop)

# YOUR CODE GOES HERE
# Compute MSE_mm
MSE_total = y_err.T @ y_err
MSE_mm = MSE_total[0][0]/y_err.shape[0]
print("MSE (matrix multiplication) = ", MSE_mm)

# YOUR CODE GOES HERE
# Compute MSE_np
MSE_np = np.mean(np.square(y_err))
print("MSE (Numpy) = ", MSE_np)