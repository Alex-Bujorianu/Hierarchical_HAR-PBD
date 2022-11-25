import json
from json import JSONDecodeError
from Baseline.utils import gauss_noise, cropping
import os
import numpy as np
import csv
import pandas as pd
from helper import create_mapping, window, convert_windowed_array_to_shape, \
    rebalance_classes, convert_windowed_Y_to_shape, convert_to_angles
import matplotlib.pyplot as plt

labels_csv = pd.read_csv("EmoPainAtHomeFull/labels.csv")
print("Label numbers in order: ", sorted(list(create_mapping(labels_csv).values())))

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
            "Walking": "Walking exercise"
        }
        return mappings[activity_name]
    except KeyError:
        # No need for mapping
        return activity_name

def get_all_data(folderpath: str) -> (np.ndarray, np.ndarray):
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
    X = np.empty(shape=(0, 6, 3), dtype=object)
    Y = []
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
                    columns = np.empty(shape=(0, 3), dtype=object)
                    for bone in bones_we_need:
                        reader = csv.reader(open(it.path + "/" + "Positions_" +  bone + ".csv"))
                        next(reader)
                        arr = list(reader)
                        remove_empty_string(arr)
                        # This typecasting is necessary
                        float_arr = [[float(y) for y in x] for x in arr]
                        arr = np.array(float_arr, dtype=object)
                        # Can't hstack, so vstack then reshape
                        columns = np.vstack((columns, arr))
                    columns = columns.reshape(-1, 6, 3)
                    X = np.vstack((X, columns))
                    append_value_multiple_times(arr=Y, label=activity_num,
                                                n_times=columns.shape[0])

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
    return (X, Y)

X = np.arange(1, 13)
print("Final result: ", window(X, 4, 1, 0.5))
X = np.arange(1, 17)
print("Another test: ", window(X, 4, 1, 0.5))
X, Y = get_all_data("EmoPainAtHomeFull")
print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)
X = window(X, 3, 40, overlap=0.5)
#X = X.reshape(-1, 120, 6, 3)
X = convert_windowed_array_to_shape(X)
print("X after windowing ", X.shape)
Y = window(Y, 3, 40, overlap=0.5)
Y = convert_windowed_array_to_shape(Y)
Y = convert_windowed_Y_to_shape(Y)
print("Y shape after windowing ", Y.shape)
X_train, Y_train, X_test, Y_test = rebalance_classes(X, Y, split_ratio=0.8, overlap_ratio=0.5)
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

# Jitter and crop training data training data
X_jitter = gauss_noise(X_train, 5)
X_cropped = cropping(X_train, 0.1)
X_train = np.concatenate((X_train, X_jitter, X_cropped), axis=0)
print(X_train.shape)
# Triplicate labels
Y_train = np.concatenate((Y_train, Y_train, Y_train), axis=0)
print(Y_train.shape)

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
np.save(arr=X_train, file="Data/X_train_full")
np.save(arr=Y_train, file="Data/Y_train_full")
np.save(arr=X_test, file="Data/X_test_full")
np.save(arr=Y_test, file="Data/Y_test_full")