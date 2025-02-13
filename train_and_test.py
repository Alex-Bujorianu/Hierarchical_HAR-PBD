import json
import numpy as np
from Baseline.utils import gauss_noise, cropping
from datetime import datetime
import bisect
import random
from helper import rebalance_classes

X = json.load(open("angle_data.json", "r"))
Y = json.load(open("label_data.json", "r"))
# How many windows in X?
window_count = 0
for key, value in X.items():
    window_count += np.array(value).shape[0]
print("Window count ", window_count)

# Use first six patients for train, remaining for test
patients = np.unique(np.array([x.split("_")[0] for x in list(X.keys())]))
print("Patients: ", patients)
print("No of unique patients ", len(patients))
n_keys_p96 = 0
keys_that_should_have = []
for key in X.keys():
    if key.split("_")[0] == '96':
        n_keys_p96 += 1
        keys_that_should_have.append(key[3:])
print("Patient 96 should have ", n_keys_p96, " keys")
# Patient no. 63 had only 5 sensors record data for each session
# therefore this patient is absent from the data
# RightLowerLeg is the missing sensor
def train_test_split(X: dict):
    print("Keys of X: ", X.keys())
    X_train = {k:{} for k in patients[0:6]}
    X_test = {k:{} for k in patients[6:]}
    # print("Empty X_train ", X_train)
    # print("Empty X_test ", X_test)
    # Python is buggy so I had to code the following workaround
    # See: https://github.com/python/cpython/issues/99932
    for key, value in X.items():
        patient = key.split("_")[0]
        if patient in X_train:
            X_train[patient][key[3:]] = value
        elif patient in X_test:
            # print("Patient nr is ", patient, " date is", key[3:])
            X_test[patient][key[3:]] = value
            # print("Keys of patient: ", X_test[patient].keys())
    return (X_train, X_test)

X_train, X_test = train_test_split(X)
window_count = 0
for key, value in X_train.items():
    for k, v in value.items():
        window_count += np.array(v).shape[0]
for key, value in X_test.items():
    for k, v in value.items():
        window_count += np.array(v).shape[0]
print("Patient's actual keys:", X_test['96'].keys())
print("Patient's intended keys: ", keys_that_should_have)
print("Number of keys of patient 96", len(X_test['96'].keys()))
assert len(X_test['96'].keys()) == n_keys_p96
print("Window count in X train", window_count)
assert window_count == 8364
Y_train = dict.fromkeys(patients[0:6], {})
Y_test = dict.fromkeys(patients[6:], {})
for key, value in Y.items():
    for patient in patients[0:6]:
        if patient == key.split("_")[0]:
            Y_train[patient].update({key[3:]: value['activity']})
    for patient in patients[6:]:
        if patient == key.split("_")[0]:
            Y_test[patient].update({key[3:]: value['activity']})


# print("Y train has ", len(list(Y_train.keys())), " keys")
# print("X train keys: ", X_train.keys())
# print("X train[39] keys: ", X_train['39'].keys())
# print("X train[48] keys ", X_train['48'].keys())
# print("Y train[39]", Y_train['39'].keys())
def print_in_common(X, Y):
    for key, value in X.items():
        for othr_key, val in value.items():
            if othr_key in Y[key]:
                print("Key in common: ", othr_key)

#print_in_common(X_test, Y_test)
# The activity label timestamps don’t match the X timestamps
# Matching by seconds is too exact
# There shouldn’t be duplicates because Python enforces uniqueness
# on dictionary keys
def truncate_seconds(input: dict) -> dict:
    input_copy = input.copy()
    for key, value in input_copy.items():
        input_copy[key] = {k[:-2]:v for (k,v) in value.items()}
    return input_copy

def typecast_to_int(timestamp: str) -> int:
    datetime_object = datetime.strptime(timestamp, '%Y_%m_%d_%H_%M')
    return datetime_object.timestamp()

def sort_dict(input_dict: dict) -> dict:
    to_return = input_dict.copy()
    for key, val in input_dict.items():
        new_dict = {typecast_to_int(k):v for (k,v) in val.items()}
        for othrkey, othrval in val.items():
            to_return[key] = dict(sorted(new_dict.items()))
    return to_return

#print(Y_train.keys())
#print("Sorted dict: ", sort_dict(Y_train)['39'])

def calculate_time(var: dict, start_position: float) -> float:
    list_of_keys = list(var.keys())
    for i in range(len(list_of_keys)-1):
        if list_of_keys[i] == start_position:
            return list_of_keys[i+1] - list_of_keys[i]

def find_nearest_key(var: dict, time: float):
    list_of_keys = list(var.keys())
    if time >= list_of_keys[-1]:
        return list_of_keys[-1]
    for i in range(len(list_of_keys)-1):
        if (time>=list_of_keys[i]) and (time<=list_of_keys[i+1]):
            distance_to_Xt = abs(time-list_of_keys[i])
            distance_to_Xt_1 = abs(time-list_of_keys[i+1])
            if distance_to_Xt < distance_to_Xt_1:
                return list_of_keys[i]
            else:
                return list_of_keys[i+1]

    # case in which given key is less than smallest key available
    return list_of_keys[0]


assert calculate_time(sort_dict(Y_train)['39'], 1616605980) == 60

def keyshift(dictionary, key, diff):
    if key in dictionary:
        token = object()
        keys = [token]*(diff*-1) + sorted(dictionary) + [token]*diff
        newkey = keys[keys.index(key)+diff]
        if newkey is token:
            return None
        else:
            return newkey
    else:
        return 'Key not found'

def get_data(X_train: dict, X_test: dict, Y_train: dict, Y_test: dict):
    #@return: returns a tuple of 4 numpy arrays
    # there are 6 sensors but only 4 angles
    X_train_numpy = np.empty((0, 120, 4, 3))
    Y_train_numpy = np.empty((0, 1))
    X_test_numpy = np.empty((0, 120, 4, 3))
    Y_test_numpy = np.empty((0, 1))
    # X_train = truncate_seconds(X_train)
    # X_test = truncate_seconds(X_test)
    # Y_train = truncate_seconds(Y_train)
    # Y_test = truncate_seconds(Y_test)
    X_train = sort_dict(X_train)
    X_test = sort_dict(X_test)
    Y_train = sort_dict(Y_train)
    Y_test = sort_dict(Y_test)
    some_times = list(Y_test['96'].keys())
    for i in range(1, len(some_times)):
        if (some_times[i] - some_times[i-1]) > 60:
            print("Timestep was greater than a minute")
    # print("Y train sorted dictionary ", Y_train)
    # print("X test sorted dict, inner dict, keys only ", X_test['95'].keys())
    for key, value in Y_train.items():
        Y_keys = []
        X_keys = []
        Y_iter = iter(value)
        for othr_key, val in value.items():
            nearest_key = find_nearest_key(X_train[key], othr_key)
            # print("Nearest X train key ", nearest_key)
            # print("Y key: ", othr_key)
            frames_within_current_key = int((othr_key - nearest_key)*40)
            # Distance should be positive
            # If Y_time is EARLIER than X_time, skip ahead
            if frames_within_current_key < 0:
                print("Time was negative: ", frames_within_current_key/40,
                      " Activity was ", val)
                continue
            X_keys.append(nearest_key)
            Y_keys.append(othr_key)
            X_session = np.array(X_train[key][nearest_key])
            # print("Shape of X_session ", X_session.shape)
            next_key = keyshift(value, othr_key, 1)
            length_of_step = 0
            if next_key is not None:
                length_of_step = next_key - othr_key
            # print("Length of step: ", length_of_step)
            # sampling rate is 40hz
            # It is assumed that the sensor data recording is longer than the activity
            # so there are multiple activities within one session
            #print(int(frames_within_current_key+(length_of_step*40)))
            # array is multi-dimensional
            indices_1 = divmod(frames_within_current_key, 120)
            indices_2 = divmod(frames_within_current_key+
                               int((length_of_step*40)), 120)
            # print("Indices 1 and 2: ", indices_1, indices_2)
            # Select only windows
            little_bit = X_session[indices_1[0]:indices_2[0]]
            # print("Shape of little bit: ", little_bit.shape)
            # print("Little bit: ", little_bit)
            # print("Shape of X_train before", X_train_numpy.shape)
            X_train_numpy = np.vstack((X_train_numpy, little_bit))
            # print("Shape of X_train after", X_train_numpy.shape)
            Y_train_numpy = np.concatenate((Y_train_numpy, np.repeat(
                np.array(val),
                little_bit.shape[0]).reshape(-1, 1)), axis=0)
    for key, value in Y_test.items():
        Y_keys = []
        X_keys = []
        for othr_key, val in value.items():
            nearest_key = find_nearest_key(X_test[key], othr_key)
            # print("Nearest key: ", nearest_key)
            frames_within_current_key = int((othr_key - nearest_key)*40)
            # Distance should be positive
            # If Y_time is EARLIER than X_time, skip ahead
            if frames_within_current_key < 0:
                continue
            X_keys.append(nearest_key)
            Y_keys.append(othr_key)
            # print("Y keys list: ", Y_keys)
            X_session = np.array(X_test[key][nearest_key])
            next_key = keyshift(value, othr_key, 1)
            length_of_step = 0
            if next_key is not None:
                length_of_step = next_key - othr_key
            #print("Length of activity ", length_of_step)
            # sampling rate is 40hz
            # It is assumed that the sensor data recording is longer than the activity
            # so there are multiple activities within one session
            # array is multi-dimensional
            indices_1 = divmod(frames_within_current_key, 120)
            indices_2 = divmod(frames_within_current_key+
                               int((length_of_step*40)), 120)
            # Select only windows
            little_bit = X_session[indices_1[0]:indices_2[0]]
            X_test_numpy = np.vstack((X_test_numpy, little_bit))
            Y_test_numpy = np.concatenate((Y_test_numpy, np.repeat(
                np.array(val),
                little_bit.shape[0]).reshape(-1, 1)), axis=0)
    return (X_train_numpy, Y_train_numpy, X_test_numpy, Y_test_numpy)

# def rebalance_classes(X_train: np.ndarray, Y_train: np.ndarray,
#                       X_test: np.ndarray, Y_test: np.ndarray, split_ratio=0.8):
#     "Rebalance classes between train and test"
#     #@param split_ratio: the fraction that goes into the training test. 80% by default.
#     X = np.concatenate((X_train, X_test), axis=0)
#     Y = np.concatenate((Y_train, Y_test), axis=0)
#     new_X_train = np.empty(shape=(0, X_train.shape[1], X_train.shape[2],
#                                   X_train.shape[3]))
#     new_X_test = np.empty(shape=(0, X_train.shape[1], X_train.shape[2],
#                                   X_train.shape[3]))
#     new_Y_train = np.empty(shape=(0, Y_train.shape[1]))
#     new_Y_test = np.empty(shape=(0, Y_train.shape[1]))
#     for i in range(20, Y.shape[0], 20):
#         # iterate through windows, append 20 windows at a time
#         # 20 windows = 1 minute, minimum length of an activity
#         if random.random() < split_ratio:
#             new_X_train = np.concatenate((new_X_train, X[i-20:i, :, :, :].reshape((20, 120, 4, 3))),
#                                          axis=0)
#             new_Y_train = np.vstack((new_Y_train, Y[i-20:i].reshape((20, 1))))
#         else:
#             new_X_test = np.vstack((new_X_test, X[i-20:i, :, :, :].reshape((20, 120, 4, 3))))
#             new_Y_test = np.vstack((new_Y_test, Y[i-20:i].reshape((20, 1))))
#     return (new_X_train, new_Y_train, new_X_test, new_Y_test)

data_tuple = get_data(X_train, X_test, Y_train, Y_test)
X_train = data_tuple[0]
Y_train = data_tuple[1]
X_test = data_tuple[2]
Y_test = data_tuple[3]
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)
X_train, Y_train, X_test, Y_test = rebalance_classes(X, Y)

print("Shape of X train ", X_train.shape)
print("Shape of Y train ", Y_train.shape)
print("Shape of X test ", X_test.shape)
print("Shape of Y test ", Y_test.shape)

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
np.save(arr=X_train, file="Data/X_train_old_organisation")
np.save(arr=Y_train, file="Data/Y_train_old_organisation")
np.save(arr=X_test, file="Data/X_test_old_organisation")
np.save(arr=Y_test, file="Data/Y_test_old_organisation")