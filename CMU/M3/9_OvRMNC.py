import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def visualize(xdata, index, title=""):
    image = xdata[index,:].reshape(20,20).T
    plt.figure()
    plt.imshow(image,cmap = "binary")
    plt.axis("off")
    plt.title(title)
    plt.show()


x_train = np.load("w3-hw3-train_x.npy")
y_train = np.load("w3-hw3-train_y.npy")
x_test = np.load("w3-hw3-test_x.npy")
y_test = np.load("w3-hw3-test_y.npy")

visualize(x_train,1234)


def ovr_model(x, y):
    model = OneVsRestClassifier(LogisticRegression()).fit(x, y)
    return model


def mn_model(x, y):
    model = LogisticRegression(max_iter=300).fit(x, y)
    return model


ovr_model = ovr_model(x_train, y_train)
mn_model = mn_model(x_train, y_train)


def accuracy(model, x, y):
    preds = model.predict(x)
    accuracy = np.sum(preds == y) / len(y) * 100
    return accuracy


ovr_training_accuracy = accuracy(ovr_model, x_train, y_train)
ovr_testing_accuracy = accuracy(ovr_model, x_test, y_test)
mn_training_accuracy = accuracy(mn_model, x_train, y_train)
mn_testing_accuracy = accuracy(mn_model, x_test, y_test)
print("OvR Training Accuracy: ", ovr_training_accuracy)
print("OvR Testing  Accuracy: ", ovr_testing_accuracy)
print("Multinomial Training Accuracy: ", mn_training_accuracy)
print("Multinomial Testing  Accuracy: ", mn_testing_accuracy)