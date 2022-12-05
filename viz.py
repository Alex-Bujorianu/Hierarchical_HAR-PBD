import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import json
from sklearn.metrics import ConfusionMatrixDisplay
from helper import max_scale
Y_train = np.load("Data/Y_train_pain.npy")
Y_test = np.load("Data/Y_test_pain.npy")

def create_dictionary(arr: np.ndarray) -> dict:
    labels = {}
    for i in range(arr.shape[0]):
        if arr[i, 0] in labels:
            labels[arr[i, 0]] = labels[arr[i, 0]] + 1
        else:
            labels[arr[i, 0]] = 1
    return labels

train_labels = np.unique(Y_train, return_counts=True)
test_labels = np.unique(Y_test, return_counts=True)
plt.bar(train_labels[0], train_labels[1])
plt.title("Label distribution of training set")
plt.xticks(list(train_labels[0]))
plt.show()
plt.bar(test_labels[0], test_labels[1])
plt.title("Label distribution of test set")
plt.xticks(list(test_labels[0]))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()
labels_csv = pd.read_csv("EmoPainAtHomeFull/labels.csv")
def create_mapping(data: pd.DataFrame) -> dict:
    mappings = {}
    labels = data['Activity'].to_list()
    codes = data['Activity_recoded'].to_list()
    for i in range(len(labels)):
        mappings[labels[i]] = codes[i]
    return mappings

mappings = create_mapping(labels_csv)
print(mappings)
print("Sorted labels ", sorted(list(mappings.values())))

# conf_matrix_cfcc = np.array(json.load(open("Results/Experiment_cfcc_pain", "r"))['Confusion matrix'])
# print("Length of conf matrix cfcc ", len(conf_matrix_cfcc[0]))
# conf_matrix_catloss = np.array(json.load(open("Results/Experiment_cat_loss_pain", "r"))['Confusion matrix'])
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_cfcc, display_labels=np.array(list(range(1, 28))))
# fig, ax = plt.subplots(figsize=(16, 16))
# plt.rcParams.update({'font.size': 16})
# disp.plot(ax=ax)
def find_key_from_value(to_search: dict, searchkey):
    for key, value in to_search.items():
        if value==searchkey:
            return key
conversion_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                   9: 8, 11: 8, 14: 9, 17: 10, 18: 11, 20: 8,
                   21: 12, 22: 13, 23: 14, 24: 15, 25: 16, 26: 17, 27: 18}
conversion_dict_reversed = {k:[] for k in conversion_dict.values()}
for key, value in conversion_dict.items():
    conversion_dict_reversed[value].append(key)

print(conversion_dict_reversed)
labels_dict = {}
for key, value in conversion_dict_reversed.items():
    for keys in value:
        labels_dict[key] = find_key_from_value(mappings, keys)
print(labels_dict)
conf_matrix= np.array(json.load(open("Results/Experiment_cfcc_pain", "r"))['Confusion matrix'])
print("Length of conf_matrix", len(conf_matrix))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.array(list(range(1, 19))))
fig, ax = plt.subplots(figsize=(16, 16))
plt.legend()
plt.rcParams.update({'font.size': 16})
disp.plot(ax=ax)
plt.show()

def sample_every_n(arr, n=10, m=10):
    to_return = []
    for i in range(0, arr.shape[0], n):
        for j in range(0, arr.shape[1], m):
            to_return.append(arr[i, j])
    return to_return

# Plot the angles, do they have the same scale?
X_train = np.load("Data/X_train_pain.npy")
for i in range(X_train.shape[2]):
    for j in range(X_train.shape[3]):
        X_train[:, :, i, j] = max_scale(X_train[:, :, i, j])
for i in range(X_train.shape[2]):
    for j in range(X_train.shape[3]):
        plt.plot(sample_every_n(X_train[:, :, i, j], n=100), label='Sensor ' + str(i+1) +
                                            ' dimension ' + str(j+1))
    plt.legend(loc='right')
    plt.show()

# As far as I can tell, they all have the same scale
# But no substitute for experimentation