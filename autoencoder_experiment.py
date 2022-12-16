from using_pretrained_Alex import autofeats_extract
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from helper import merge_walking, pick_labels
import matplotlib.pyplot as plt


def print_scores(Y_test, predictions):
    print("Accuracy: ", accuracy_score(Y_test, predictions))
    print("F1: ", f1_score(Y_test, predictions, average="weighted"))
    print("Precision: ", precision_score(Y_test, predictions, average='weighted'))
    print("Recall: ", recall_score(Y_test, predictions, average="weighted"))

def new_encoding(arr: np.ndarray):
    conversion_dict = {1: 1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7,
                       9:8, 10:11, 11:8, 14:9, 17:10, 18:11, 20:8,
                       21:12, 22:13, 23:14, 24:15, 25:16, 26:17, 27:18}
    for i in range(arr.shape[0]):
        arr[i][0] = conversion_dict[arr[i][0]]

X_train = np.load("Data/X_train_6s_40hz_pain.npy")
X_test = np.load("Data/X_test_6s_40hz_pain.npy")
Y_train = np.load("Data/Y_train_6s_40hz_pain.npy")
Y_test = np.load("Data/Y_test_6s_40hz_pain.npy")
new_encoding(Y_train)
new_encoding(Y_test)
# Merge walking should be called after new encoding
merge_walking(Y_train, Y_test)
to_predict = [12, 15, 1, 8, 2, 3, 14, 11, 10]
X_train, X_test, Y_train, Y_test = pick_labels(to_predict=to_predict, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
embeddings_train, activities_train = autofeats_extract(X_train, "autoencoder.hdf5")
print("Embeddings train: ", embeddings_train[0:10, :])
embeddings_test, activities_test = autofeats_extract(X_test, "autoencoder.hdf5")
print(type(embeddings_train))
print(embeddings_train.shape)
# encoder = LabelBinarizer()
# encoder.fit(Y_train)
# Y_train = encoder.transform(Y_train)
# encoder.fit(Y_test)
# Y_test = encoder.transform(Y_test)
print("Y train (sanity check): ", Y_train[0:5])
print("Shapes of Y train and Y test: ", Y_train.shape, Y_test.shape)

neural_network = MLPClassifier(hidden_layer_sizes=(80, 40, 20, 10), batch_size=200, solver='adam',
                               activation='identity', learning_rate_init=0.001, learning_rate='invscaling', power_t=0.3, verbose=True)
neural_network.fit(embeddings_train, Y_train)
print(neural_network.out_activation_)
print("Train predictions (sanity check): ", neural_network.predict(embeddings_train))
predictions = neural_network.predict(embeddings_test)
print("Shape of predictions: ", predictions.shape)
print("Predictions: ", predictions[0:10])
# Y_test = encoder.inverse_transform(Y_test)
# predictions = encoder.inverse_transform(predictions)
print_scores(Y_test, predictions)
conf_matrix = confusion_matrix(Y_test, predictions)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()

