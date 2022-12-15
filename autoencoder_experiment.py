from using_pretrained_Alex import autofeats_extract
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix
from helper import merge_walking


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
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
embeddings_train, activities_train = autofeats_extract(X_train, "autoencoder.hdf5")
embeddings_test, activities_test = autofeats_extract(X_test, "autoencoder.hdf5")
print(type(embeddings_train))
print(embeddings_train.shape)
encoder = LabelBinarizer()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)
encoder.fit(Y_test)
Y_test = encoder.transform(Y_test)
print("Shapes of Y train and Y test: ", Y_train.shape, Y_test.shape)

neural_network = MLPClassifier(hidden_layer_sizes=(20, 20, 20), batch_size=200, solver='sgd',
                               activation='relu', learning_rate_init=0.01, learning_rate='invscaling', power_t=0.3, verbose=True)
neural_network.fit(embeddings_train, Y_train)
print("Train predictions (sanity check): ", neural_network.predict(embeddings_train))
predictions = neural_network.predict(embeddings_test)
print("Shape of predictions: ", predictions.shape)
print("Predictions: ", predictions[0:10, :])
print_scores(Y_test, predictions)

