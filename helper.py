import numpy as np
import pandas as pd
import random
import csv
from angleforAlex import get_half_skel_joint_angles
import json
from json import JSONDecodeError
import os

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
        # print("Indices: ", indices)
        data_dict[label] = {'X': X[indices], 'Y': Y[indices]}
    #print("Data dict: ", data_dict)
    X_train = np.empty(shape=(0, X.shape[1], X.shape[2],
                                  X.shape[3]))
    X_test = np.empty(shape=(0, X.shape[1], X.shape[2],
                                  X.shape[3]))
    Y_train = np.empty(shape=(0, Y.shape[1]))
    Y_test = np.empty(shape=(0, Y.shape[1]))
    if not (20/(1-overlap_ratio)).is_integer():
        raise ValueError("Overlap ratio is not compatible with window size")
    minute = int(20 / (1-overlap_ratio))
    for key, value in data_dict.items():
        X_train = np.vstack((X_train,
                             value['X'][0:int(split_ratio*value['X'].shape[0])]))
        X_test = np.vstack((X_test,
                             value['X'][int(split_ratio * value['X'].shape[0]):]))
        #print("Subset of Y train: ", value['Y'][0:int(split_ratio * value['Y'].shape[0])])
        Y_train = np.vstack((Y_train, value['Y'][0:int(split_ratio * value['Y'].shape[0])]))
        Y_test = np.vstack((Y_test,
                            value['Y'][int(split_ratio * value['Y'].shape[0]):]))
        #print("Y_test in for loop", Y_test)
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
