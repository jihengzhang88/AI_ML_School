import numpy as np
import util
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

sys.path.append('../logreg_stability')
### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1


def plot_decision_boundary(clf, x, y, filename):
    plt.figure()
    # Plot the points with different colors for each class
    plt.scatter(x[y == 0][:, 1], x[y == 0][:, 2], label='Class 0', alpha=0.5)
    plt.scatter(x[y == 1][:, 1], x[y == 1][:, 2], label='Class 1', alpha=0.5)

    # Plot the decision boundary
    x_values = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    y_values = -(clf.theta[0] + clf.theta[1] * x_values) / clf.theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary', color='red')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Decision Boundary and Data Points')
    plt.savefig(filename)
    plt.show()


def compute_metrics(clf, x, y):
    y_pred = clf.predict(x)
    y_pred = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm.ravel()
    A0 = TN / (TN + FP)
    A1 = TP / (TP + FN)
    return accuracy, balanced_accuracy, A0, A1


def upsample_minority(x, y, kappa):
    """Upsample the minority class examples in the training set."""
    # Identify the minority class
    minority_class = 1 if np.sum(y) / y.size < 0.5 else 0
    majority_class = 1 - minority_class

    # Separate the data into majority and minority classes
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the number of times to repeat the minority class
    repeat_factor = int(1 / kappa)

    # Upsample the minority class
    x_minority_upsampled = np.repeat(x[minority_indices], repeat_factor, axis=0)
    y_minority_upsampled = np.repeat(y[minority_indices], repeat_factor, axis=0)

    # Combine with the majority class
    x_upsampled = np.vstack((x[majority_indices], x_minority_upsampled))
    y_upsampled = np.hstack((y[majority_indices], y_minority_upsampled))

    return x_upsampled, y_upsampled


def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_vanilla = save_path.replace(WILDCARD, 'vanilla')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    x_train, y_train = util.load_dataset(train_path)
    x_val, y_val = util.load_dataset(validation_path)

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_vanilla using np.savetxt()
    clf_vanilla = LogisticRegression()
    clf_vanilla.fit(x_train, y_train)
    y_pred_naive = clf_vanilla.predict(x_val)
    np.savetxt(output_path_vanilla, y_pred_naive)

    # Compute metrics
    accuracy, balanced_accuracy, A0, A1 = compute_metrics(clf_vanilla, x_val, y_val)

    print(f'Overall Vanilla Accuracy: {accuracy:.4f}')
    print(f'Balanced Vanilla Accuracy: {balanced_accuracy:.4f}')
    print(f'Vanilla Accuracy for Class 0 (A0): {A0:.4f}')
    print(f'Vanilla Accuracy for Class 1 (A1): {A1:.4f}')

    # Plot decision boundary
    plot_decision_boundary(clf_vanilla, x_val, y_val, 'decision_boundary_vanilla.png')

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    x_train_upsampled, y_train_upsampled = upsample_minority(x_train, y_train, kappa)
    clf_upsampling = LogisticRegression()
    clf_upsampling.fit(x_train_upsampled, y_train_upsampled)
    y_pred_upsampling = clf_upsampling.predict(x_val)
    np.savetxt(output_path_upsampling, y_pred_upsampling)

    # Compute metrics
    accuracy, balanced_accuracy, A0, A1 = compute_metrics(clf_upsampling, x_val, y_val)

    print(f'Overall Upsampling Accuracy: {accuracy:.4f}')
    print(f'Balanced Upsampling Accuracy: {balanced_accuracy:.4f}')
    print(f'Upsampling Accuracy for Class 0 (A0): {A0:.4f}')
    print(f'Upsampling Accuracy for Class 1 (A1): {A1:.4f}')

    # Plot decision boundary
    plot_decision_boundary(clf_upsampling, x_val, y_val, 'decision_boundary_upsampling.png')
    # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
