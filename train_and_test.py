import json
import numpy as np
from Baseline.utils import gauss_noise
from datetime import datetime

X = json.load(open("angle_data.json", "r"))
Y = json.load(open("label_data.json", "r"))


# Use first six patients for train, remaining for test
patients = np.unique(np.array([x.split("_")[0] for x in list(X.keys())]))
print("No of unique patients ", len(patients))
# Patient no. 63 had only 5 sensors record data for each session
# therefore this patient is absent from the data
# RightLowerLeg is the missing sensor
def train_test_split(X: dict):
    X_train = dict.fromkeys(patients[0:6], {})
    X_test = dict.fromkeys(patients[6:], {})
    for key, value in X.items():
        for patient in patients[0:6]:
            if patient == key.split("_")[0]:
                X_train[patient].update({key[3:]: value})
        for patient in patients[6:]:
            if patient == key.split("_")[0]:
                X_test[patient].update({key[3:]: value})
    return (X_train, X_test)

X_train, X_test = train_test_split(X)
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
print("Sorted dict: ", sort_dict(Y_train)['39'])

def calculate_time(var: dict, start_position: float) -> float:
    # Calculates how long an activity lasts in continuous time
    # for a single step (because steps are uneven)
    # So just the difference between key_n and key_n+1
    list_of_keys = list(var.keys())
    for i in range(len(list_of_keys)-1):
        if list_of_keys[i] == start_position:
            return list_of_keys[i+1] - list_of_keys[i]

def find_nearest_key(var: dict, time: float):
    list_of_keys = list(var.keys())
    for i in range(len(list_of_keys)):
        # label time may be a few seconds earlier than sensor recording time
        # so added a bit of fuzziness
        if time>=list_of_keys[i]-3:
            return list_of_keys[i]
        else:
            return list_of_keys[0]


assert calculate_time(sort_dict(Y_train)['39'], 1616605980) == 60

def get_data(X_train: dict, X_test: dict, Y_train: dict, Y_test: dict):
    #@return: returns a tuple of 4 numpy arrays
    # there are 6 sensors but only 4 angles
    X_train_numpy = np.empty((0, 120, 4, 3))
    Y_train_numpy = np.empty((0, 120))
    X_test_numpy = np.empty((0, 120, 4, 3))
    Y_test_numpy = np.empty((0, 120))
    # X_train = truncate_seconds(X_train)
    # X_test = truncate_seconds(X_test)
    # Y_train = truncate_seconds(Y_train)
    # Y_test = truncate_seconds(Y_test)
    X_train = sort_dict(X_train)
    X_test = sort_dict(X_test)
    Y_train = sort_dict(Y_train)
    Y_test = sort_dict(Y_test)
    for key, value in Y_train.items():
        frames_within_current_key = 0
        X_keys = []
        for othr_key, val in value.items():
            print("Val: ", val)
            #print("Key and other key: ", key, " ", othr_key)
            #print("Value shape: ", val.shape)
            print("X train keys: ", list(X_train[key].keys()))
            print("Othr key: ", othr_key)
            nearest_key = find_nearest_key(X_train[key], othr_key)
            print("Nearest key: ", nearest_key)
            X_keys.append(nearest_key)
            print("X keys list: ", X_keys)
            X_session = np.array(X_train[key][nearest_key])
            # Turns out each activity lasts for a minute
            length_of_step = 60
            #print("Length of activity ", length_of_step)
            # sampling rate is 40hz
            # It is assumed that the sensor data recording is longer than the activity
            print("Frames within current key ", frames_within_current_key)
            print(frames_within_current_key + (length_of_step * 40))
            little_bit = X_session[frames_within_current_key:int(frames_within_current_key+(length_of_step*40))]
            X_train_numpy = np.vstack((X_train_numpy, little_bit))
            frames_within_current_key += int(length_of_step * 40)
            # Key has changed? Reset frame count
            if nearest_key != X_keys[-1]:
                frames_within_current_key = 0
            Y_train_numpy = np.concatenate((Y_train_numpy, np.repeat(
                np.array(val),
                little_bit.shape[0] * little_bit.shape[1]).reshape(-1, 120)), axis=0)
    for key, value in X_test.items():
        #print(value.keys())
        for othr_key, val in value.items():
            val = np.array(val)
            if othr_key in Y_test[key]:
                X_test_numpy = np.vstack((X_test_numpy, val))
                Y_test_numpy = np.concatenate((Y_test_numpy, np.repeat(
                    np.array(Y_test[key][othr_key]),
                    val.shape[0] * val.shape[1]).reshape(-1, 120)), axis=0)
    return (X_train_numpy, Y_train_numpy, X_test_numpy, Y_test_numpy)

data_tuple = get_data(X_train, X_test, Y_train, Y_test)
print("Shape of X train ", data_tuple[0].shape)
print("Shape of Y train ", data_tuple[1].shape)
print("Shape of X test ", data_tuple[2].shape)
print("Shape of Y test ", data_tuple[3].shape)
X_train = data_tuple[0]

# Jitter training data
X_train = np.concatenate((X_train, gauss_noise(X_train, 5)), axis=0)
# Angles can be negative after jittering
# so add 360 or make it 0, or keep it negative
# Nadia says 0 is the best

# There are many negative numbers
def make_positive(input_arr: np.ndarray) -> np.ndarray:
    for iy, ix, iz, iw in np.ndindex(input_arr.shape):
        # if input_arr[iy, ix, iz, iw] < 0:
        #     print("Negative number found")
        input_arr[iy, ix, iz, iw] = max(input_arr[iy, ix, iz, iw], 0)


make_positive(X_train)
print(X_train.shape)
