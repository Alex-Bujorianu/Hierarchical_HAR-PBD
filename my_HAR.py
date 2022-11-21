from Baseline.HierarchicalHAR_PBD import build_model
import numpy as np
import Baseline.utils as utils
import numpy as np
from collections import Counter
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # enable a clean command line  window. The thing is to define os. before importing tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # use GPU with ID=0.
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



HARmodel = HAR_model_wrapper(adjacency_matrix=adjacency_matrix,
                             timestep=120, node_num=4, feature_num=3)

X_train = np.load("Data/X_train.npy")
Y_train = np.load("Data/Y_train.npy")
X_test = np.load("Data/X_test.npy")
Y_test = np.load("Data/Y_test.npy")
result = np.matmul(norm_adj, X_train[0, 0, :, :])
print("Result of matrix multiplication of normalized adjacency matrix with "
      "4x3 matrix from X[0, 0]: ", result,
      "and its shape: ", result.shape)

#Tensorflow expects classes to start from 0, otherwise it throws a fit
Y_train = Y_train - 1
Y_test = Y_test - 1
print("Classes in Y_train: ", np.unique(Y_train))

def train_model(model: HAR_model_wrapper, X_train: np.ndarray, X_test: np.ndarray,
                Y_train: np.ndarray, Y_test: np.ndarray):
    AdjNorm = utils.MakeGraph(model.adjacency_matrix)
    print("Shape of normalised adjacency matrix: ", AdjNorm.shape)
    graphtrain = utils.my_combine(AdjNorm, X_train)
    graphtest = utils.my_combine(AdjNorm, X_test)
    print("Shape of Y train before one-hot encoding: ", Y_train.shape)
    # One hot encoding
    Y_train = to_categorical(Y_train, num_classes=model.num_classes)
    Y_test = to_categorical(Y_test, num_classes=model.num_classes)
    print("Shape of categorically encoded Y_train: ", Y_train.shape)
    print("Shape of categorically encoded Y_test: ", Y_test.shape)
    model.model.compile(optimizer=Adam(learning_rate=5e-4, decay=1e-5),
                  loss={
                        'HARout': 'categorical_crossentropy'
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


train_model(HARmodel, X_train, X_test, Y_train, Y_test)