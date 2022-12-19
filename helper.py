import numpy as np
import pandas as pd
import random
import csv
from angleforAlex import get_half_skel_joint_angles
import json
from json import JSONDecodeError
import os
from statistics import mean

def new_encoding(arr: np.ndarray):
    conversion_dict = {1: 1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7,
                       9:8, 10:11, 11:8, 14:9, 17:10, 18:11, 20:8,
                       21:12, 22:13, 23:14, 24:15, 25:16, 26:17, 27:18}
    for i in range(arr.shape[0]):
        arr[i][0] = conversion_dict[arr[i][0]]


def merge_walking(Y_train, Y_test, Y_validation=None):
    merge_walking = {14: 14, 18: 14}
    for i in range(Y_train.shape[0]):
        if (Y_train[i][0] == 18) or (Y_train[i][0] == 14):
            Y_train[i][0] = merge_walking[Y_train[i][0]]

    for i in range(Y_test.shape[0]):
        if (Y_test[i][0] == 18) or (Y_test[i][0] == 14):
            Y_test[i][0] = merge_walking[Y_test[i][0]]

    if Y_validation is not None:
        for i in range(Y_validation.shape[0]):
            if (Y_validation[i][0] == 18) or (Y_validation[i][0] == 14):
                Y_validation[i][0] = merge_walking[Y_validation[i][0]]

def pick_labels(to_predict, X_train, X_test, Y_train, Y_test, Y_validation=None, X_validation=None):
    # Get indices of X&Y
    # If you pass one, you have to pass both
    if (Y_validation is not None) or (X_validation is not None):
        assert X_validation is not None
        assert Y_validation is not None
    indices_train = []
    indices_test = []
    indices_validation = []
    for activity_label in to_predict:
        for index in np.where(Y_train[:, 0] == activity_label)[0]:
            indices_train.append(index)
        for index in np.where(Y_test[:, 0] == activity_label)[0]:
            indices_test.append(index)
        if Y_validation is not None:
            for index in np.where(Y_validation[:, 0] == activity_label)[0]:
                indices_validation.append(index)

    # Subset
    X_train_new = X_train[indices_train]
    X_test_new = X_test[indices_test]
    Y_train_new = Y_train[indices_train]
    Y_test_new = Y_test[indices_test]
    if Y_validation is not None:
        X_validation_new = X_validation[indices_validation]
        Y_validation_new = Y_validation[indices_validation]
        return (X_train_new, X_test_new, Y_train_new, Y_test_new, X_validation_new, Y_validation_new)
    return (X_train_new, X_test_new, Y_train_new, Y_test_new)

def rolling_mean(arr, ratio) -> np.ndarray:
    "Arr should be 1D array-like"
    assert type(ratio) == int
    to_return = []
    for i in range(ratio, len(arr)+1, ratio):
        to_return.append(mean(arr[i-ratio:i]))
    return np.array(to_return, dtype=float)

def downsample(X, input_freq=40, output_freq=10) -> np.ndarray:
    if not (input_freq/output_freq).is_integer():
        raise TypeError("This function only works when the downsampling ratio is an integer")
    ratio = int(input_freq/output_freq)
    X_downsampled = np.empty(shape=(X.shape[0], int(X.shape[1]/ratio), X.shape[2], X.shape[3]))
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            for z in range(X.shape[3]):
                X_downsampled[i, :, j, z] = rolling_mean(X[i, :, j, z],
                                                  ratio=ratio)

    return X_downsampled


def max_scale(arr: np.ndarray) -> np.ndarray:
    #@param arr: a 2D array of shape (windows, window_length)
    # Apply this to each node/angle
    new_arr = np.empty(shape=arr.shape)
    maximum = np.max(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i, j] = arr[i, j] / maximum
    return new_arr

def max_scale_all(X_train, X_test, X_validation=None):
    for i in range(X_train.shape[2]):
        for j in range(X_train.shape[3]):
            X_train[:, :, i, j] = max_scale(X_train[:, :, i, j])
    for i in range(X_test.shape[2]):
        for j in range(X_test.shape[3]):
            X_test[:, :, i, j] = max_scale(X_test[:, :, i, j])
    if X_validation is not None:
        for i in range(X_validation.shape[2]):
            for j in range(X_validation.shape[3]):
                X_validation[:, :, i, j] = max_scale(X_validation[:, :, i, j])


# Labels 9, 11 and 20 are basically the same thing
# 21 and 22 (vacuuming and vacuuming car) are also similar
# Painting shelves and painting wall?
def merge_option_1(Y: np.ndarray):
    washing_machine = set([9, 11, 20])
    for i in range(Y.shape[0]):
        label = Y[i][0]
        if label in washing_machine:
            Y[i][0] = 9
        if (label == 12) or (label == 13):
            Y[i][0] = 12

def merge_option_2(Y: np.ndarray) -> np.ndarray:
    washing_machine = set([9, 11, 20])
    for i in range(Y.shape[0]):
        label = Y[i][0]
        if label in washing_machine:
            Y[i][0] = 9
        if (label == 12) or (label == 13):
            Y[i][0] = 12
        # merge vacuuming
        if (label == 21) or (label == 22):
            Y[i][0] = 21

def create_mapping(data: pd.DataFrame) -> dict:
    mappings = {}
    labels = data['Activity'].to_list()
    codes = data['Activity_recoded'].to_list()
    for i in range(len(labels)):
        mappings[labels[i]] = codes[i]
    return mappings

# Need to window to convert 3D to 4D
def window(data: np.ndarray, window_time=3, sampling_rate=40, overlap=None):
    to_return = []
    step = int(window_time*sampling_rate)
    if overlap==None:
        for i in range(step, len(data)+1, step):
            to_return.append(data[i-step:i])
        return np.array(to_return)
    else:
        for i in range(step, len(data)+1, step):
            # print("i is: ", i)
            to_return.append(data[i-step:i])
        # print("To_return: ", to_return)
        overlapped_windows = [to_return[0]]
        # overlapped_windows = np.empty(shape=(0, step, data.shape[1], data.shape[2]))
        # print("Overlapped windows: ", overlapped_windows)
        partial_step = int(step*overlap)
        for i in range(step-partial_step, len(data)-partial_step, partial_step):
            # partial = int(step*overlap)
            # print(to_return[i][partial:])
            # overlapped_window = np.append(to_return[i][partial:],
            #                                to_return[i+1][0:partial])
            # overlapped_windows.append(overlapped_window)
            # print("Overlapped windows: ", overlapped_windows)
            # print("i: ", i)
            subwindow = data[i:i+step]
            # print("Subwindow: ", subwindow)
            overlapped_windows.append(subwindow)
            # print("overlapped windows: ", overlapped_windows)

        return np.array(overlapped_windows, dtype=object)

def convert_windowed_array_to_shape(arr: np.ndarray) -> np.ndarray:
    "After using the window function with overlap, it is not straightforward to reshape the array to the correct dimensions"
    # the last subwindow will likely not have correct shape because arbitrary data will not fit it
    arr_to_return = np.empty(shape=(len(arr)-1,) + arr[0].shape)
    # Do not use vstack, numpy arrays are very slow to append
    # Pre-allocate memory in one big array and index
    for i in range(len(arr_to_return)):
        arr_to_return[i] = arr[i]
    return arr_to_return

def convert_windowed_Y_to_shape(arr: np.ndarray) -> np.ndarray:
    "Instead of a label for every row in the window, 1 label at the start of the window"
    new_arr = np.empty(shape=(arr.shape[0], 1))
    for i in range(arr.shape[0]):
        new_arr[i] = arr[i, 0]
    return new_arr

def my_shuffle(arr, seed=123):
    "Shuffles in place using provided random seed"
    generator = np.random.default_rng(seed)
    generator.shuffle(arr, axis=0)

# Note: random sampling is NOT necessarily the best approach here
# We want classes in the train set to be represented in the test set
# Sampling at random introduces large differences in the distribution of the two sets
# Although the dataset is large, the classes are many and unbalanced
# Thus, some classes end up completely unrepresented in the smaller test set
# Solution: stratified sampling
def rebalance_classes(X: np.ndarray, Y: np.ndarray, split_ratio=0.8, overlap_ratio=0):
    "Rebalance classes between train and test"
    #@param split_ratio: the fraction that goes into the training test. 80% by default.
    labels = np.unique(Y)
    data_dict = dict.fromkeys(labels)
    for label in labels:
        indices = np.where(Y==label)[0]
        data_dict[label] = {'X': X[indices], 'Y': Y[indices]}
    # An activity in the training set may be done by 1 patient
    # whereas the same activity in the test might be done by a different patient
    # so we have to sample from different ‘zones’ in the array, by shuffling
    # But, the order in X must correspond to the order in Y
    # and the order of the frames within the windows must remain unchanged
    for key, value in data_dict.items():
        my_shuffle(value['X'])
        my_shuffle(value['Y'])
    X_train = np.empty(shape=(0, X.shape[1], X.shape[2],
                                  X.shape[3]))
    X_test = np.empty(shape=(0, X.shape[1], X.shape[2],
                                  X.shape[3]))
    Y_train = np.empty(shape=(0, Y.shape[1]))
    Y_test = np.empty(shape=(0, Y.shape[1]))
    if not (20/(1-overlap_ratio)).is_integer():
        raise ValueError("Overlap ratio is not compatible with window size")
    for key, value in data_dict.items():
        X_train = np.vstack((X_train,
                             value['X'][0:int(split_ratio*value['X'].shape[0])]))
        X_test = np.vstack((X_test,
                             value['X'][int(split_ratio * value['X'].shape[0]):]))
        #print("Subset of Y train: ", value['Y'][0:int(split_ratio * value['Y'].shape[0])])
        Y_train = np.vstack((Y_train, value['Y'][0:int(split_ratio * value['Y'].shape[0])]))
        Y_test = np.vstack((Y_test,
                            value['Y'][int(split_ratio * value['Y'].shape[0]):]))
    return (X_train, Y_train, X_test, Y_test)


# This function is very slow
def convert_to_angles(arr: np.ndarray):
    return get_half_skel_joint_angles(arr)

def remove_empty_string(input: list) -> list:
    #@param input: a list of lists
    #@return: a list of lists with the empty string at the end of each sublist removed
    #Also removes first element
    for i in range(len(input)):
        input[i] = input[i][1:-1]

def append_value_multiple_times(arr: list, label: int, n_times: int):
    for i in range(n_times):
        arr.append(label)

def map_activity_names(activity_name: str) -> str:
    "Activity names are not consistent between the 2 datasets"
    try:
        mappings = {
            "Tidying up": "Tidying up room",
            "Changing bedsheets": "Changing bed sheets",
            "Walking - dogs": "Walking dogs",
            "Vacuuming - car": "Vacuuming (car)",
            "Unloading dishwasher": "Unloading dish washer",
            "Dusting - car": "Dusting (car)",
            "Loading and unloading washing machine": "Loading & unloading washing machine",
            "Walking": "Walking exercise",
            "Loading dishwasher": "Loading dish washer",
            "Washing Up": "Washing up"
        }
        return mappings[activity_name]
    except KeyError:
        # No need for mapping
        return activity_name


def get_all_data(folderpath: str, time=3, sampling_rate=40) -> (np.ndarray, np.ndarray):
    "This function returns windowed X and Y"
    labels_csv = pd.read_csv("EmoPainAtHomeFull/labels.csv")
    labels = create_mapping(labels_csv)
    print("Labels: ", labels)
    bones_we_need = [
      "ChestBottom",
      "RightThigh",
      "RightUpperArm",
      "RightLowerLeg",
      "RightForeArm",
      "Hip"
   ]
    X = np.empty(shape=(0, time*sampling_rate, 6, 3), dtype=object)
    Y = []
    frame_count = 0
    for it in os.scandir(folderpath):
        if it.is_dir():
            try:
                metadata = json.load(open(it.path + "/meta.json"))
                # we need all 6
                bones = metadata['bones']
                if all(x in bones for x in bones_we_need):
                    path_array = str(it.path).split("/")
                    activity_name = path_array[1].split("_")[1]
                    activity_name = map_activity_names(activity_name)
                    try:
                        activity_num = labels[activity_name]
                    except KeyError:
                        print("Unknown activity ", activity_name, " encountered, skipping…")
                        continue
                    # Figure out the number of rows in this session
                    # Fortunately, it’s in the json
                    columns = np.empty(shape=(metadata['frame_count'], 0), dtype=object)
                    for bone in bones_we_need:
                        reader = csv.reader(open(it.path + "/" + "Positions_" +  bone + ".csv"))
                        next(reader)
                        arr = list(reader)
                        remove_empty_string(arr)
                        # This typecasting is necessary
                        float_arr = [[float(y) for y in x] for x in arr]
                        arr = np.array(float_arr, dtype=object)
                        columns = np.hstack((columns, arr))
                    columns = columns.reshape(-1, 6, 3)
                    frame_count += columns.shape[0]
                    columns_windowed = window(columns, time, sampling_rate, overlap=None)
                    # Temi thinks we should window each activity instance
                    # to prevent overlapping
                    #print("Shape of windowed columns: ", columns_windowed.shape)
                    X = np.vstack((X, columns_windowed))
                    append_value_multiple_times(Y, activity_num,
                                n_times=columns_windowed.shape[0] * columns_windowed.shape[1])

            except JSONDecodeError:
                print("Decode error encountered in ", it.path + "/meta.json", " skipping…")
                continue
            except FileNotFoundError:
                print("File not found, likely that a sensor failed. Skipping…")
                continue
            except StopIteration:
                print("Oopsie, ", it.path + "/" + "Positions_" +  bone + ".csv file must have been empty. Skipping…")
                continue
    Y = np.array(Y, dtype=object)
    Y = window(Y, time, sampling_rate)
    Y = convert_windowed_Y_to_shape(Y)
    print("Total number of frames: ", frame_count)
    return (X, Y)
