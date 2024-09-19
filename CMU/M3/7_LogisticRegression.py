import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, c, title="", xlabel="$x_1$", ylabel="$x_2$", classes=["", ""], alpha=1):
    N = len(c)
    colors = ['royalblue', 'crimson']
    symbols = ['o', 's']

    plt.figure(figsize=(5, 5), dpi=120)

    for i in range(2):
        x = data[:, 0][c == i]
        y = data[:, 1][c == i]

        plt.scatter(x, y, color=colors[i], marker=symbols[i], edgecolor="black", linewidths=0.4, label=classes[i],
                    alpha=alpha)

    plt.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title(title)


def plot_contour(w):
    res = 500
    vals = np.linspace(-0.05, 1.05, res)
    x, y = np.meshgrid(vals, vals)
    XY = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    prob = sigmoid(map_features(XY) @ w.reshape(-1, 1))
    pred = np.round(prob.reshape(res, res))
    plt.contour(x, y, pred)


train = np.load("w3-hw1-data-train.npy")
test = np.load("w3-hw1-data-test.npy")
train_data, train_gt = train[:,:2], train[:,2]
test_data, test_gt = test[:,:2], test[:,2]
format = dict(xlabel="x-velocity m/s", ylabel="y-velocity, m/s", classes=["0 - Missed Target","1 - Hit Target"])
plot_data(train_data, train_gt, **format)


# YOUR CODE GOES HERE
def sigmoid(h):
    return 1/(1 + np.exp(-h))


def map_features(data):
    x = data[:, 0]
    y = data[:, 1]
    feature = []
    for i in range(data.shape[0]):
        row = []
        for p in range(9):
            for q in range(p+1):
                row.append((x[i] ** (p-q)) * (y[i] ** q))
        feature.append(row)

    return np.array(feature)


def loss(data, y, w):
    wt_x = map_features(data) @ w
    J1 = -np.log(sigmoid(wt_x)) * y
    J2 = -np.log(1 - sigmoid(wt_x)) * (1-y)
    L = np.sum(J1 + J2)
    return L


def gradloss(data, y, w):
    mapped_data = map_features(data)
    wt_x = mapped_data @ w
    err = sigmoid(wt_x) - y
    gradloss = mapped_data.T @ err
    return gradloss


def grad_desc(data, y, w0=np.zeros(45), iterations=100, stepsize=0.1):
    w = w0
    for i in range(iterations):
       w = w - stepsize * gradloss(data, y, w)
    return w


w = grad_desc(train_data, train_gt)

# YOUR CODE GOES HERE (loss plot, print w)
test1 = sigmoid(map_features(train_data) @ w)
preds_train = np.round(sigmoid(map_features(train_data) @ w)).astype(int)
preds_test = np.round(sigmoid(map_features(test_data) @ w)).astype(int)

accuracy_train = np.sum(preds_train == train_gt) / len(train_gt) * 100
accuracy_test = np.sum(preds_test == test_gt) / len(test_gt) * 100

print("          w = ", w)
print("True Train Classes: ", train_gt.astype(int))
print("Train Predictions: ", preds_train)
print("True Test Classes: ", test_gt.astype(int))
print("Test Predictions: ", preds_test)

print("Train Accuracy: ", accuracy_train, "%")
print("Test Accuracy: ", accuracy_test, "%")

# Plot data and decision boundary
plot_data(train_data, train_gt, **format)
plot_contour(w)
plt.show()
