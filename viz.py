import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

Y_train = np.load("Data/Y_train.npy")
Y_test = np.load("Data/Y_test.npy")

print("Shape of Y_train: ", Y_train.shape)

def create_dictionary(arr: np.ndarray) -> dict:
    labels = {}
    for i in range(arr.shape[0]):
        if arr[i, 0] in labels:
            labels[arr[i, 0]] = labels[arr[i, 0]] + 1
        else:
            labels[arr[i, 0]] = 1
    return labels

train_labels = create_dictionary(Y_train)
test_labels = create_dictionary(Y_test)

plt.bar(*zip(*train_labels.items()))
plt.title("Label distribution of training set")
plt.xticks(list(train_labels.keys()))
plt.show()
plt.bar(*zip(*test_labels.items()))
plt.title("Label distribution of test set")
plt.xticks(list(test_labels.keys()))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()