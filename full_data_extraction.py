import json
from json import JSONDecodeError
from Baseline.utils import gauss_noise, cropping
import os
import numpy as np
import csv
import pandas as pd
from helper import create_mapping, window, convert_windowed_array_to_shape, \
    rebalance_classes, convert_windowed_Y_to_shape, convert_to_angles, get_all_data
import matplotlib.pyplot as plt

labels_csv = pd.read_csv("EmoPainAtHomeFull/labels.csv")
print("Label numbers in order: ", sorted(list(create_mapping(labels_csv).values())))


X = np.arange(1, 13)
print("Final result: ", window(X, 4, 1, 0.5))
X = np.arange(1, 17)
print("Another test: ", window(X, 4, 1, 0.5))
X, Y = get_all_data("EmoPainAtHomeFull", time=3, sampling_rate=40)
# Healthy participants were sampled at 10Hz
# Just train on sick participants first
# X_healthy, Y_healthy = get_all_data("EmoPainHealthy")
# X = np.concatenate((X, X_healthy), axis=0)
# Y = np.concatenate((Y, Y_healthy), axis=0)
print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)
# Performance is bad with 50% overlap
X_train, Y_train, X_test, Y_test = rebalance_classes(X, Y,
                        split_ratio=0.8, overlap_ratio=0)
# Sanity check, does the distribution between train/test look reasonable?
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

print("Shape of Y_train: ", Y_train.shape)
print("Shape of Y_test: ", Y_test.shape)

plt.bar(*zip(*train_labels.items()))
plt.title("Label distribution of training set")
plt.xticks(list(train_labels.keys()))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()
plt.bar(*zip(*test_labels.items()))
plt.title("Label distribution of test set")
plt.xticks(list(test_labels.keys()))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()

# This stage will take a while
X_train = convert_to_angles(X_train)
X_test = convert_to_angles(X_test)

# Jitter and crop training data
X_jitter, Y_jitter = gauss_noise(X_train, 5, labels=[4, 6, 7, 18, 22], Y=Y_train)
list_of_indices = []
for label in [4, 6, 7, 18, 22]:
    indices = np.where(Y_train==label)[0]
    for index in indices:
        list_of_indices.append(index)

X_cropped = cropping(X_train[list_of_indices, :, :, :], 0.1)
X_train = np.concatenate((X_train, X_jitter, X_cropped), axis=0)
print(X_train.shape)
Y_train = np.concatenate((Y_train, Y_jitter, Y_jitter), axis=0)
print(Y_train.shape)
train_labels = create_dictionary(Y_train)
plt.bar(*zip(*train_labels.items()))
plt.title("Label distribution of training set after augmentation")
plt.xticks(list(train_labels.keys()))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()

# Angles can be negative after jittering
# so add 360 or make it 0, or keep it negative
# Nadia says 0 is the best

print("Unique activities in train ", np.unique(Y_train))
print("Unique activities in test ", np.unique(Y_test))

# There are many negative numbers
def make_positive(input_arr: np.ndarray) -> np.ndarray:
    for iy, ix, iz, iw in np.ndindex(input_arr.shape):
        # if input_arr[iy, ix, iz, iw] < 0:
        #     print("Negative number found")
        input_arr[iy, ix, iz, iw] = max(input_arr[iy, ix, iz, iw], 0)


make_positive(X_train)

# Save
np.save(arr=X_train, file="Data/X_train_pain_3s_resampled")
np.save(arr=Y_train, file="Data/Y_train_pain_3s_resampled")
np.save(arr=X_test, file="Data/X_test_pain_3s_resampled")
np.save(arr=Y_test, file="Data/Y_test_pain_3s_resampled")