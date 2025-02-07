# Code Author: Guanting Cen, Chongyang Wang (My PhD was supported by UCL Overseas Research Scholarship and Graduate Research Scholarship, 
# and partially by the EU Future and Emerging Technologies (FET) Proactive Programme H2020-EU.1.2.2 (Grant agreement 824160; EnTimeMent).

# Revision Date: 26 12 2021


import numpy as np
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # enable a clean command line  window. The thing is to define os. before importing tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # use GPU with ID=0.
import tensorflow as tf
import keras
import warnings
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
from collections import Counter


# Segmentation(Sliding Window)
def segmentation(mat, overlap_ratio, window_len):
    x_segment = np.zeros((1, 66))
    y_segment = np.array([[0, 0]])
    m = len(mat)  # number of rows
    n = len(mat[0])  # number of columns
    step = int(window_len * (1 - overlap_ratio))

    length = 0  # length of the data
    num_frames = 0  # number of frames
    # if remain length of matrix is samller than a window length, the last window padding with 0
    while (m - length) > window_len:
        x_segment = np.concatenate((x_segment, mat[num_frames * step:num_frames * step + window_len, :66]), 0)
        y_HAR_segment = int(Counter(mat[num_frames * step:num_frames * step + window_len, 70]).most_common(1)[0][0])
        # standing, sitting, walking and others all considered as transition
        # number of classes of PBD is 6, including 5 activities and 1 transition
        if y_HAR_segment == 6 or y_HAR_segment == 7 or y_HAR_segment == 8:
            y_HAR_segment = 0
        y_PDB_segment = int(Counter(mat[num_frames * step:num_frames * step + window_len, 72]).most_common(1)[0][0])

        y_segment = np.concatenate((y_segment, [[y_HAR_segment, y_PDB_segment]]), 0)
        num_frames += 1
        length = 90 * (num_frames + 1)

        # padding last window with 0
    len_remain = m - length
    mat_padding = np.zeros([window_len - len_remain, n])
    mat_last = np.concatenate((mat[length:, :], mat_padding[:, :]), 0)

    # final concatenation
    x_segment = np.concatenate((x_segment, mat_last[:, :66]), 0)
    y_HAR_segment = int(Counter(mat_last[:, 70]).most_common(1)[0][0])
    if y_HAR_segment == 6 or y_HAR_segment == 7 or y_HAR_segment == 8:
        y_HAR_segment = 0
    y_PDB_segment = int(Counter(mat_last[:, 72]).most_common(1)[0][0])
    y_segment = np.concatenate((y_segment, [[y_HAR_segment, y_PDB_segment]]), 0)

    # reshape the data
    num_frames = num_frames + 1
    x_segment = np.reshape(x_segment[1:], (num_frames, window_len, 66))
    y_segment = y_segment[1:]

    return x_segment, y_segment

#jittering(Gaussian Noise)
def gauss_noise(data, dev, labels=None, Y=None):
    "Pass the list of labels you want to selectively augment along with Y"
    # dev is deviation
    if labels is not None:
        assert Y is not None
        new_data = np.empty(shape=(0, data.shape[1], data.shape[2], data.shape[3]))
        new_Y = np.empty(shape=(0, 1))
        for label in labels:
            indices = np.where(Y==label)[0]
            new_data = np.concatenate((new_data, data[indices, :, :, :]), axis=0)
            new_Y = np.concatenate((new_Y, Y[indices, :]), axis=0)
        noise = np.random.normal(0, dev, new_data.shape)
        jittered_data = new_data + noise
        return (jittered_data, new_Y)
    else:
        noise = np.random.normal(0, dev, data.shape)
        data_jitternig = data + noise
        return data_jitternig

def calculate_degree_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    assert  adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    degree_matrix = np.zeros((adjacency_matrix.shape[0], adjacency_matrix.shape[1]))
    for i,j in np.ndindex(adjacency_matrix.shape):
        degree_matrix[i, i] = np.sum(np.array(adjacency_matrix[i, :]))
    return degree_matrix


def MakeGraph(adjacency_matrix):
# This function is used for the situation where the target graph can be easily & manually defined.
# Here is an example how we define the graph of human skeleton with 22 nodes.
 # 1. define the 22X22 adjacency matrix .
    Adj = adjacency_matrix
    # Adj = np.zeros((22, 22))
    # Adj[0, 1] = 1
    # Adj[0, 4] = 1
    # Adj[0, 7] = 1
    # Adj[1, 0] = 1
    # Adj[1, 2] = 1
    # Adj[2, 1] = 1
    # Adj[2, 3] = 1
    # Adj[3, 2] = 1
    # Adj[4, 0] = 1
    # Adj[4, 5] = 1
    # Adj[5, 4] = 1
    # Adj[5, 6] = 1
    # Adj[6, 5] = 1
    # Adj[7, 0] = 1
    # Adj[7, 8] = 1
    # Adj[8, 7] = 1
    # Adj[8, 9] = 1
    # Adj[8, 14] = 1
    # Adj[8, 19] = 1
    # Adj[9, 8] = 1
    # Adj[9, 10] = 1
    # Adj[10, 9] = 1
    # Adj[10, 11] = 1
    # Adj[11, 10] = 1
    # Adj[11, 12] = 1
    # Adj[12, 11] = 1
    # Adj[12, 13] = 1
    # Adj[13, 12] = 1
    # Adj[14, 8] = 1
    # Adj[14, 15] = 1
    # Adj[15, 14] = 1
    # Adj[15, 16] = 1
    # Adj[16, 15] = 1
    # Adj[16, 17] = 1
    # Adj[17, 16] = 1
    # Adj[17, 18] = 1
    # Adj[18, 17] = 1
    # Adj[19, 8] = 1
    # Adj[19, 20] = 1
    # Adj[20, 19] = 1
    # Adj[20, 21] = 1
    # Adj[21, 20] = 1
    #print(Adj)
# 2. define the diagonal 22X22 degree matrix.
#     Degree = np.zeros((22, 22))
#     Degree[0, 0] = 3
#     Degree[1, 1] = 2
#     Degree[2, 2] = 2
#     Degree[3, 3] = 1
#     Degree[4, 4] = 2
#     Degree[5, 5] = 2
#     Degree[6, 6] = 1
#     Degree[7, 7] = 2
#     Degree[8, 8] = 4
#     Degree[9, 9] = 2
#     Degree[10, 10] = 2
#     Degree[11, 11] = 2
#     Degree[12, 12] = 2
#     Degree[13, 13] = 1
#     Degree[14, 14] = 2
#     Degree[15, 15] = 2
#     Degree[16, 16] = 2
#     Degree[17, 17] = 2
#     Degree[18, 18] = 1
#     Degree[19, 19] = 2
#     Degree[20, 20] = 2
#     Degree[21, 21] = 1
#     assert Degree.all() == calculate_degree_matrix(Adj).all()
#     print("Degree matrix: \n", Degree)
#     print("Calculated degree matrix: \n", calculate_degree_matrix(Adj))

# 3. compute and output the normalized adjacency matrix, referring to Equation 1 in the paper.
    Degree = calculate_degree_matrix(Adj)
    I = np.identity(adjacency_matrix.shape[0])
    Degree_hat = Degree + I
    Adj_hat = Adj + I
    DegreePower = fractional_matrix_power(Degree_hat, -0.5)
    AdjNorm = np.matmul(np.matmul(DegreePower, Adj_hat), DegreePower)

    return AdjNorm

def crop(dimension, start, end):
    # Thanks to the nice person named marc-moreaux on Github page:https://github.com/keras-team/keras/issues/890
    # who created this beautiful and sufficient function: ).
    # This function is not used for this paper, but I really like it so pasted here for your possible use.
    # Crops (or slices) a Tensor on a given dimension from start to end.
    # example : to crop tensor x[:, :, 5:10],
    # call slice(2, 5, 10) as you want to crop on the second dimension.
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

# I don't understand why there are 2 cropping functions
#Cropping
def cropping(data,prob):
    #prob is the probability of dropping
    time_step = data.shape[1]
    node_num = data.shape[2]
    num_samples = data.shape[0]

    for i in range(num_samples):
        drop_samples = np.random.randint(0, time_step, int(round(time_step*prob)))
        for j in drop_samples:
            drop_nodes = np.random.randint(0, node_num, int(round(node_num*prob)))
            for k in drop_nodes:
                data[i,j,k] = 0
                data[i,j,k+node_num] = 0
                data[i,j,k+node_num*2] = 0
    return data

def focal_loss(weights=None, gamma=None, num_class=None):
# the focal loss.
    if weights is None:
        weights = [1.0, 1.0 ] # for binary task. change to [1.0,1.0,1.0,....] for multi-class tasks.
    if gamma is None:
        gamma = 1.0
    if num_class is None:
        num_class = 2 # for binary task. change to num_class for multi-class tasks.
    def focaloss(y_true,y_pred):
        weights1 = tf.cast(weights, dtype=tf.float32)
        weights1 = tf.expand_dims(weights1, 0)
        weights1 = tf.tile(weights1, [tf.shape(y_true)[0], 1]) * y_true
        weights1 = tf.reduce_sum(weights1, axis=1)
        weights1 = tf.expand_dims(weights1, 1)
        weights1 = tf.tile(weights1, [1, num_class])
        # y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)  # some implementations performed this step.
        CE = -y_true * tf.math.log(y_pred)  #the categorical cross-entropy loss.
        modulator = (1 - y_true * y_pred) ** gamma  # This is what shall be, according to the definition of Focal loss in the original paper.
        # modulator = tf.exp(-gamma * y_true * y_pred - gamma * tf.math.log1p(tf.exp(-1.0 * y_pred)))  # however, this is also seen in other implementations.
        loss = modulator * CE
        weighted_loss = weights1 * loss
        FL = tf.reduce_sum(weighted_loss)
        FL /= tf.reduce_sum(y_true)
        return FL
    return focaloss

def class_balance_weights(beta, sample_per_class):
# the class-balanced term that we combined with the focal loss in our paper, which is computed before hand and fixed during each training fold.
    en = 1.0 - np.power(beta, sample_per_class)
    alpha = (1 - beta) / np.array(en)
    alpha = alpha / np.sum(alpha)
    return alpha

# here is how CFCC loss is called for binary PBD, hyper-parameters of beta and gamma should be tuned.
#model.compile(optimizer=Adam(learning_rate=0.0005),
#                            loss=[focal_loss(weights=class_balance_weights(0.9999,[np.sum(y_train[:, 0]), np.sum(y_train[:, 1])]),
#                            gamma=2, num_class=2)],
#                            metrics=['categorical_accuracy'])

def build_callbacks(modelname,person):
# modelname is a string indicating the name of current training period.
# person is a constant stating the participant ID that used by me, given the LOSO is used as the validation method.
# both of them are used to name the hdf5 files.

    # callback 1: Save the model with improved results after each epoch,
    metricmonitor = 'val_PBDout_categorical_accuracy'
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelname + 'Best' + person + '.hdf5',
                                                   monitor=metricmonitor, verbose=1,
                                                   save_best_only=True)

    # callback 2: Stop if Acc=1
    class EarlyStoppingByValAcc(keras.callbacks.Callback):
        def __init__(self, monitor, value, verbose):
            super(keras.callbacks.Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            if current == self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                    self.model.stop_training = True
    callbacks = [EarlyStoppingByValAcc(monitor=metricmonitor, value=1.0000, verbose=1),
                           checkpointer
                            # TensorBoard(log_dir='dir/logfolder') # Add yours to check the TensorBoard
                          ]
    return  callbacks

def my_combine(AdjNorm, X):
    #@param AdjNorm: the normalized adjacency matrix
    #@param X: the B X T X node_num X dimensions array
    # For the EmoPain@Home dataset, node_num=4 angles and they have 3 dimensions
    "Basically, matrix multiply the last 2 dimensions in X with the norm adj matrix"
    X_copy = X.copy()
    assert AdjNorm.shape[1] == X.shape[2]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result = np.matmul(AdjNorm, X[i, j, :, :])
            X_copy[i, j, :, :] = result
    return X_copy

def combine(Adj, X, time_step, node_num, feature_num):
# Transfer the input matrix X (NxTxD) into graph sequences (N x T x node_num x feature_num) as input
# assume X=[X_coordinate, Y_coordinate, Z_coordinate].
# Adj is Adjacency matrix of (None, node_num, node_num).
# time_step is the length of each data segment.
# node_num is the number of nodes/joints, feature_num is the feature dimension per node/joint which is 3 with three-axis coordinate.
    X_coordinate = X[:, :, 0:node_num]
    Y_coordinate = X[:, :, node_num:2*node_num]
    Z_coordinate = X[:, :, 2*node_num:3*node_num]
    num_sample, _, _ = X.shape

    X_coordinate = np.reshape(X_coordinate, (num_sample, time_step, node_num, 1))
    Y_coordinate = np.reshape(Y_coordinate, (num_sample, time_step, node_num, 1))
    Z_coordinate = np.reshape(Z_coordinate, (num_sample, time_step, node_num, 1))
    nodefeature = np.concatenate((X_coordinate, Y_coordinate, Z_coordinate), axis=-1)

    list = np.arange(0, num_sample, 1)
    list1 = np.arange(0, time_step, 1)
    buffer = np.zeros((node_num, feature_num))
    cobuffer = np.zeros((num_sample, time_step, node_num, feature_num))

    for index in range(len(list)):
        for index1 in range(len(list1)):
            buffer = nodefeature[index, index1, :, :]
            cobuffer[index, index1, :, :] = np.matmul(Adj, buffer)

    return cobuffer

# here is how the callbacks and input transformation are used.
#AdjNorm = MakeGraph()
#graphtrain = combine(AdjNorm, X_train, time_step, node_num, feature_num)
#graphvalid = combine(AdjNorm, X_valid, time_step, node_num, feature_num)

#model.fit(graphtrain,
#       y_train,
#       batch_size=batchsize,
#       epochs=epoch,
#       callbacks=build_callbacks(modelname, person),
#       validation_data=(graphvalid, y_valid))
