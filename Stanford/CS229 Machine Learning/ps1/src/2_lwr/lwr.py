import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    # Predict on evaluation set
    y_pred = clf.predict(x_eval)
    # Get MSE value on the validation set
    mse = np.mean((y_pred - y_eval) ** 2)
    print(f'MSE on validation set: {mse:.4f}')
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    plt.scatter(x_train[:, 1], y_train, color='blue', marker='x', label='Training data')
    # plt.scatter(x_eval[:, 1], y_eval, color='green', marker='o', label='Validation data') # Uncomment to show validation set
    plt.scatter(x_eval[:, 1], y_pred, color='red', marker='o', label='Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Locally Weighted Linear Regression')
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)
        for i in range(m):
            w = self._get_weights(x[i])
            theta = np.linalg.inv(self.x.T @ w @ self.x) @ (self.x.T @ w @ self.y)
            y_pred[i] = x[i] @ theta
        return y_pred
        # *** END CODE HERE ***

    def _get_weights(self, x_i):
        """Compute the weights for a given input x_i."""
        m = self.x.shape[0]
        w = np.exp(-np.sum((self.x - x_i) ** 2, axis=1) / (2 * self.tau ** 2))
        return np.diag(w)

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
