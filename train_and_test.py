import json
import numpy as np

X = json.load(open("angle_data.json", "r"))
Y = json.load(open("label_data.json", "r"))


# Use first six patients for train, remaining for test
patients = np.unique(np.array([x.split("_")[0] for x in list(X.keys())]))
print(patients)
print("No of unique patients ", len(patients))
# Patient no. 63 had only 5 sensors record data for each session
# therefore this patient is absent from the data
# RightLowerLeg is the missing sensor
def train_test_split(X: dict):
    X_train = dict.fromkeys(patients[0:6], {})
    X_test = dict.fromkeys(patients[6:], {})
    print("Length of X keys ", len(X.keys()))
    print("X keys: ", X.keys())
    for key, value in X.items():
        print("key train_test_split: ", key[3:])
        for patient in patients[0:6]:
            if patient == key.split("_")[0]:
                X_train[patient].update({key[3:]: value})
        for patient in patients[6:]:
            if patient == key.split("_")[0]:
                X_test[patient].update({key[3:]: value})
    return (X_train, X_test)

X_train, X_test = train_test_split(X)
#print(X_train['39'])
print("X train has ", len(X_train.keys()), " keys")
print("X test has ", len(X_test.keys()), " keys")
Y_train = dict.fromkeys(patients[0:6], {})
Y_test = dict.fromkeys(patients[6:], {})
for key, value in Y.items():
    for patient in patients[0:6]:
        if patient == key.split("_")[0]:
            Y_train[patient].update({key[3:]: value['activity']})
    for patient in patients[6:]:
        if patient == key.split("_")[0]:
            Y_test[patient].update({key[3:]: value['activity']})


print("Y train has ", len(list(Y_train.keys())), " keys")
print("X train keys: ", X_train.keys())
print("X train[39] keys: ", X_train['39'].keys())
print("X train[48] keys ", X_train['48'].keys())
print("Y train[39]", Y_train['39'].keys())
def print_in_common(X, Y):
    for key, value in X.items():
        for othr_key, val in value.items():
            if othr_key in Y[key]:
                print("Key in common: ", othr_key)

print_in_common(X_test, Y_test)
print("X test keys: ", X_test.keys())
print("Y test keys: ", Y_test.keys())
print("X_test[95] keys ", X_test['95'].keys())
print("Y_test[95] keys ", Y_test['95'].keys())

def get_data(X_train: dict, X_test: dict, Y_train: dict, Y_test: dict):
    #@return: returns a tuple of 4 numpy arrays
    # there are 6 sensors but only 4 angles
    X_train_numpy = np.empty((0, 120, 4, 3))
    Y_train_numpy = np.empty((0, 120))
    X_test_numpy = np.empty((0, 120, 4, 3))
    Y_test_numpy = np.empty((0, 120))
    for key, value in X_train.items():
        print(value.keys())
        for othr_key, val in value.items():
            #print("Key and other key: ", key, " ", othr_key)
            val = np.array(val)
            #print("Value shape: ", val.shape)
            if othr_key in Y_train[key]:
                X_train_numpy = np.vstack((X_train_numpy, val))
                Y_train_numpy = np.concatenate((Y_train_numpy, np.repeat(
                    np.array(Y_train[key][othr_key]),
                    val.shape[0] * val.shape[1]).reshape(-1, 120)), axis=0)
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
