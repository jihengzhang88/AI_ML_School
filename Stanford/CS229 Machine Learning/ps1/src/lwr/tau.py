import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth parameter tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_tau = None
    best_mse = float('inf')

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_valid)
        mse = np.mean((y_pred - y_valid) ** 2)
        print(f'Tau: {tau}, MSE on validation set: {mse:.4f}')

        # Plot data
        plt.scatter(x_train[:, 1], y_train, color='blue', marker='x', label='Training data')
        plt.scatter(x_valid[:, 1], y_pred, color='red', marker='o', label='Predictions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Tau: {tau}, MSE on validation set: {mse:.4f}')
        plt.show()

        if mse < best_mse:
            best_mse = mse
            best_tau = tau

    print(f'Best tau: {best_tau}, Best MSE on validation set: {best_mse:.4f}')

    # Fit a LWR model with the best tau value
    clf = LocallyWeightedLinearRegression(best_tau)
    clf.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    y_test_pred = clf.predict(x_test)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    print(f'MSE on test set: {test_mse:.4f}')

    # Save predictions to pred_path
    np.savetxt(pred_path, y_test_pred)

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
