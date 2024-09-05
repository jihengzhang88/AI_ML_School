import numpy as np
import util
import matplotlib.pyplot as plt


def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set.
    plot_decision_boundary(clf, x_train, y_train)
    # Use save_path argument to save various visualizations for your own reference.
    y_pred = clf.predict(x_train)
    np.savetxt(save_path, y_pred)
    # *** END CODE HERE ***


def plot_decision_boundary(clf, x, y):
    """Plot data and the decision boundary.

    Args:
        clf: Trained classifier.
        x: Feature data.
        y: Labels.
    """
    # Plotting the data points
    plt.scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Set1, marker='o', edgecolor='k')

    # Plotting the decision boundary
    x_values = np.array([np.min(x[:, 1]), np.max(x[:, 1])])
    y_values = -(clf.theta[0] + clf.theta[1] * x_values) / clf.theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Decision Boundary and Data')
    plt.show()


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
    def sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-z))
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            h = self.sigmoid(np.dot(x, self.theta))
            gradient = np.dot(x.T, (h - y)) / y.size
            theta_prev = self.theta.copy()
            self.theta -= self.learning_rate * gradient

            # Convergence check
            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.eps:
                if self.verbose:
                    print(f'Converged in {i+1} iterations.')
                break

        if self.verbose:
            print(f'Final theta: {self.theta}')
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(np.dot(x, self.theta))
        # *** END CODE HERE ***


if __name__ == '__main__':
    # print('==== Training model on data set A ====')
    # main(train_path='ds1_a.csv',
    #      save_path='logreg_pred_a.txt')

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b.txt')
