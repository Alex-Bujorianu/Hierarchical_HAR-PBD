import numpy as np
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
from Baseline.HierarchicalHAR_PBD import build_model
import Baseline.utils as utils
import numpy as np
from collections import Counter
import h5py
import os
from viz import merge_option_1, merge_option_2
import tensorflow as tf
import keras
from tensorflow.keras.layers import * # for the new versions of Tensorflow, layers, models, regularizers, and optimizers shall be imported from Tensorflow.
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from keras.losses import * # and losses, metrics, callbacks, and backend can still be used from Keras directly.
from keras.metrics import *
from keras import metrics
from sklearn.metrics import *
from keras import backend as K
from keras.backend import *
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power
from keras.utils.np_utils import *
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn import metrics
import seaborn as sns
import pandas as pd
import json

adjacency_matrix = np.zeros((4, 4))
adjacency_matrix[0, 1] = 1
adjacency_matrix[1, 0] = 1
adjacency_matrix[1, 2] = 1
adjacency_matrix[1, 3] = 1
adjacency_matrix[2, 1] = 1
adjacency_matrix[2, 3] = 1
adjacency_matrix[3, 1] = 1
adjacency_matrix[3, 2] = 1
print("Adjacency matrix: ", adjacency_matrix)
norm_adj = utils.MakeGraph(adjacency_matrix)

class HAR_model_wrapper():
    "A class to hold a HAR model and its important properties"
    timestep = 0
    node_num = 0
    feature_num = 0
    adjacency_matrix = None
    num_classes = 0
    def __init__(self, adjacency_matrix, timestep, node_num, feature_num, num_class_HAR=26):
        assert adjacency_matrix.shape[0] == node_num
        assert adjacency_matrix.shape[1] == node_num
        self.model = build_model(timestep=timestep, body_num=node_num, feature_dim=feature_num,
                              gcn_units_HAR=26, lstm_units_HAR=24, adjacency_matrix=adjacency_matrix,
                              gcn_units_PBD=16, lstm_units_PBD=24,
                              num_class_HAR=num_class_HAR, num_class_PBD=2)[1]
        self.num_classes = num_class_HAR
        self.timestep = timestep
        self.node_num = node_num
        self.feature_num = feature_num
        self.adjacency_matrix = adjacency_matrix



X_train = np.load("Data/X_train.npy")
Y_train = np.load("Data/Y_train.npy")
X_test = np.load("Data/X_test.npy")
Y_test = np.load("Data/Y_test.npy")
# Option for merging. Make sure to call this before -1
merge_option_1(Y_train)
merge_option_1(Y_test)
result = np.matmul(norm_adj, X_train[0, 0, :, :])
print("Result of matrix multiplication of normalized adjacency matrix with "
      "4x3 matrix from X[0, 0]: ", result,
      "and its shape: ", result.shape)

#Tensorflow expects classes to start from 0, otherwise it throws a fit
Y_train = Y_train - 1
Y_test = Y_test - 1
print("Classes in Y_train: ", np.unique(Y_train))

class_counts = np.unique(Y_train, return_counts=True)
# Add missing labels to class_counts
def add_missing_labels(class_counts: tuple) -> tuple:
    labels = class_counts[0]
    counts = class_counts[1]
    new_labels = np.array(list(range(0, 26)))
    # Assume that the labels with 0 samples have 1 sample
    # to avoid a divide by zero error
    # the effect of this assumption is quite negligible
    new_counts = np.ones(shape=(26,))
    print(new_counts)
    res = {new_labels[i]: new_counts[i] for i in range(len(new_labels))}
    for i in range(len(labels)):
        res[labels[i]] = counts[i]
    return res

HARmodel = HAR_model_wrapper(adjacency_matrix=adjacency_matrix,
                             timestep=120, node_num=4, feature_num=3)


def train_model(model: HAR_model_wrapper, X_train: np.ndarray, X_test: np.ndarray,
                Y_train: np.ndarray, Y_test: np.ndarray):
    AdjNorm = utils.MakeGraph(model.adjacency_matrix)
    graphtrain = utils.my_combine(AdjNorm, X_train)
    graphtest = utils.my_combine(AdjNorm, X_test)
    print("Shape of X train :", X_train.shape)
    print("Shape of Y train before one-hot encoding: ", Y_train.shape)
    class_counts = np.unique(Y_train, return_counts=True)
    class_counts = add_missing_labels(class_counts)
    print("Class counts: ", class_counts)
    # One hot encoding
    Y_train = to_categorical(Y_train, num_classes=model.num_classes)
    Y_test = to_categorical(Y_test, num_classes=model.num_classes)
    print("Shape of categorically encoded Y_train: ", Y_train.shape)
    print("Shape of categorically encoded Y_test: ", Y_test.shape)
    # Beta = 0.9999 produces a really small loss which makes it hard
    # for the model to update its weights
    # Beta = 0.3
    model.model.compile(optimizer=Adam(learning_rate=5e-4, decay=1e-5),
                  loss={
                        #'HARout': 'categorical_crossentropy'
                        'HARout': utils.focal_loss(weights = utils.class_balance_weights(0.30,
                                     list(class_counts.values())),
                                     gamma=5, num_class=model.num_classes)
                        },
                  loss_weights={'HARout': 1.},
                  metrics=['categorical_accuracy'])

    model.model.fit(x=graphtrain,
              y=Y_train,
              batch_size=150,
              epochs=100,
              #callbacks=utils.build_callbacks('Model', str(valid_patient)),
              validation_data=(graphtest, Y_test)
              )
    model.model.save("Models/GC_LSTM_HAR")
    return model.model

train = input("Train model? Yes/no")
if train=="Yes":
    model = train_model(HARmodel, X_train, X_test, Y_train, Y_test)
else:
    print("Loading model…")
    model = keras.models.load_model("Models/GC_LSTM_HAR")
AdjNorm = utils.MakeGraph(HARmodel.adjacency_matrix)
graphtest = utils.my_combine(AdjNorm, X_test)
print("Y test before categorical encoding: ", Y_test.shape)
predictions = model.predict(graphtest)
print("Shape of predictions: ", predictions.shape)
print("First predictions: ", predictions[0])
# Do these numbers actually sum to 1?
for i in range(predictions.shape[0]):
    print("Sum: ", np.sum(predictions[0]))
    break

#Pretty much. So pick the most likely class
def zeros_and_ones(arr):
    to_return = np.zeros(shape=arr.shape[0])
    for i in range(arr.shape[0]):
        print("Index of biggest number: ", np.argmax(arr[i]))
        to_return[i] = np.argmax(arr[i])
    return to_return

# Transform predictions back to original shape
predictions = zeros_and_ones(predictions)
# Save results
results = {"F1 score": f1_score(Y_test, predictions, average='weighted'),
           "Accuracy": accuracy_score(Y_test, predictions),
           "Precision": precision_score(Y_test, predictions, average='weighted'),
           "Recall": recall_score(Y_test, predictions, average='weighted'),
           "Confusion matrix": confusion_matrix(Y_test, predictions)}
print("F1 score ", f1_score(Y_test, predictions, average='weighted'))
print("Accuracy ", accuracy_score(Y_test, predictions))
print("Precision ", precision_score(Y_test, predictions, average='weighted'))
print("Recall ", recall_score(Y_test, predictions, average='weighted'))
name = input("Please name this experiment")
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
json.dump(results, open("Results/Experiment_" + name, "w"), cls=NumpyEncoder)