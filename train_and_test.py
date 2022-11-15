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
    X_train = dict.fromkeys(patients[0:6], [])
    X_test = dict.fromkeys(patients[6:], [])
    for key,value in X.items():
        for patient in patients[0:6]:
            if patient == key.split("_")[0]:
                X_train[patient] = X_train[patient] + value
        for patient in patients[6:]:
            if patient == key.split("_")[0]:
                X_test[patient] = X_test[patient] + value
        return (X_train, X_test)

X_train, X_test = train_test_split(X)
print("X train has ", len(X_train.keys()), " keys")
print("X test has ", len(X_test.keys()), " keys")
Y_train = dict.fromkeys(patients[0:6], [])
print(Y_train)
Y_test = dict.fromkeys(patients[6:], [])
for key, value in Y.items():
    for patient in patients[0:6]:
        if patient == key.split("_")[0]:
            Y_train[patient].append(value['activity'])
    for patient in patients[6:]:
        if patient == key.split("_")[0]:
            Y_test[patient].append(value['activity'])

print("Y train has ", len(list(Y_train.keys())), " keys")