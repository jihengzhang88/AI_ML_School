import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_data(x, y, c,title="Phase of simulated material"):
    xlim = [0,52.5]
    ylim = [0,1.05]
    markers = [dict(marker="o", color="royalblue"), dict(marker="s", color="crimson"), dict(marker="^", color="limegreen")]
    labels = ["Solid", "Liquid", "Vapor"]

    plt.figure(dpi=150)

    for i in range(1+max(c)):
        plt.scatter(x[c==i], y[c==i], s=60, **(markers[i]), edgecolor="black", linewidths=0.4,label=labels[i])

    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Temperature, K")
    plt.ylabel("Pressure, atm")
    plt.box(True)


def plot_colors(classify, res=40):
    xlim = [0,52.5]
    ylim = [0,1.05]
    xvals = np.linspace(*xlim,res)
    yvals = np.linspace(*ylim,res)
    x,y = np.meshgrid(xvals,yvals)
    XY = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)

    color = classify(XY).reshape(res,res)

    cmap = ListedColormap(["lightblue","lightcoral","palegreen"])
    plt.pcolor(x, y, color, shading="nearest", zorder=-1, cmap=cmap,vmin=0,vmax=2)
    return


train = np.load("w3-hw2-data-train.npy")
test = np.load("w3-hw2-data-test.npy")
train_data, train_gt = train[:,:2], train[:,2].astype(int)
test_data, test_gt = test[:,:2], test[:,2].astype(int)
plot_data(train_data[:,0], train_data[:,1],train_gt)


def convert_to_binary_dataset(classes, A):
    classes_binary = (classes == A).astype(int)
    return classes_binary


# YOUR CODE GOES HERE (gradient descent and related functions)
def sigmoid(h):
    g = 1/(1 + np.exp(-h))
    return g


def gradloss(data, y, w):
    data_with_bias = np.hstack([np.ones((data.shape[0],1)),data])  # output nX3 matrix
    wt_x = data_with_bias @ w.reshape(-1, 1) # output nX1 matrix
    err = sigmoid(wt_x) - y.reshape(-1, 1) # output nX1 matrix
    gradloss = err.T @ data_with_bias
    return gradloss


def grad_desc(data, y, w0=np.array([0,0,0]), iterations=10000, stepsize=0.001):
    w = w0
    for i in range(iterations):
       w = w - stepsize * gradloss(data, y, w)
    return w.flatten()


# YOUR CODE GOES HERE (training)
def generate_ovr_ws(data, classes):
    classes_unique = np.unique(classes)
    w = []
    for A in classes_unique:
        y_binary = convert_to_binary_dataset(classes, A)
        wi = grad_desc(data, y_binary)
        w.append(wi)
    return w


ws = generate_ovr_ws(train_data, train_gt)
for i in range(len(ws)):
    print(f'Class {i}: w = {ws[i]}')


def classify(xy):
    # YOUR CODE GOES HERE
    class_probs = np.zeros((xy.shape[0], len(ws)))
    data_with_bias = np.hstack([np.ones((xy.shape[0],1)),xy])
    for i, w in enumerate(ws):
        class_probs[:, i] = data_with_bias @ w # Get the probability for each class

    # Select the class with the highest probability for each point
    predictions = np.argmax(class_probs, axis=1)

    return predictions


# YOUR CODE GOES HERE (accuracy)
def get_accuracy(data, c):
    preds = classify(data)
    accuracy = np.sum(preds == c) / len(c) * 100
    return accuracy

print("Training Accuracy: ", get_accuracy(train_data, train_gt))
print(" Testing Accuracy: ", get_accuracy(test_data, test_gt))
plot_data(train_data[:,0], train_data[:,1], train_gt)
plot_colors(classify)
plt.show()