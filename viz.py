import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import json
from sklearn.metrics import ConfusionMatrixDisplay
from helper import max_scale
Y_train_noaug = np.load("Data/Y_train_pain.npy")
Y_train_sel_aug = np.load("Data/Y_train_pain_3s_resampled.npy")
def new_encoding(arr: np.ndarray):
    conversion_dict = {1: 1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7,
                       9:8, 10:11, 11:8, 14:9, 17:10, 18:11, 20:8,
                       21:12, 22:13, 23:14, 24:15, 25:16, 26:17, 27:18}
    for i in range(arr.shape[0]):
        arr[i][0] = conversion_dict[arr[i][0]]
print("Classes before new encoding ", np.unique(Y_train_noaug))
new_encoding(Y_train_noaug)
new_encoding(Y_train_sel_aug)
print("Classes after new encoding ", np.unique(Y_train_noaug))
print("Shape of Y test ", Y_train_sel_aug.shape)
print("Shape of Y train", Y_train_noaug.shape)

def create_dictionary(arr: np.ndarray) -> dict:
    labels = {}
    for i in range(arr.shape[0]):
        if arr[i, 0] in labels:
            labels[arr[i, 0]] = labels[arr[i, 0]] + 1
        else:
            labels[arr[i, 0]] = 1
    return labels

no_aug_labels = np.unique(Y_train_noaug, return_counts=True)
sel_aug_labels = np.unique(Y_train_sel_aug, return_counts=True)
plt.bar(no_aug_labels[0], no_aug_labels[1])
plt.title("Label distribution of training set before augmentation")
plt.xticks(list(no_aug_labels[0]))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()
plt.bar(sel_aug_labels[0], sel_aug_labels[1])
plt.title("Label distribution of training set after augmentation")
plt.xticks(list(sel_aug_labels[0]))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()
labels_csv = pd.read_csv("EmoPainAtHomePain/labels.csv")
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

conf_matrix_cfcc = np.array(json.load(open("Results/Experiment_cfcc_pain", "r"))['Confusion matrix'])
print("Length of conf matrix cfcc ", len(conf_matrix_cfcc[0]))
conf_matrix_catloss = np.array(json.load(open("Results/9_labels/Experiment_9_labels_cfcc_3s_full", "r"))['Confusion matrix'])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_cfcc, display_labels=np.array(list(range(1, 19))))
fig, ax = plt.subplots(figsize=(16, 16))
plt.rcParams.update({'font.size': 16})
disp.plot(ax=ax, include_values=False)
plt.show()
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
print("Labels dict ", labels_dict)
conf_matrix= np.array(json.load(open("Results/Experiment_9_labels_cfcc_3s_40hz_pain", "r"))['Confusion matrix'])
print("Length of conf_matrix", len(conf_matrix))
original_labels = [12, 15, 1, 8, 2, 3, 14, 11, 10]
conf_matrix = conf_matrix / conf_matrix.astype(np.float).sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=[labels_dict[original_labels[i]] for i in range(len(original_labels))])
#disp.ax_.set(xlabel='Predicted label')
fig, ax = plt.subplots(figsize=(20, 20))
fig.subplots_adjust(bottom=0.2, left=0.2)
plt.legend()
plt.rcParams.update({'font.size': 16})
disp.plot(ax=ax, xticks_rotation=45)
plt.show()
# To help figure out this key
for i in range(0, 8):
    print("Label nr ", i, "corresponds to ", labels_dict[original_labels[i]])

# Let’s create a readable bar chart
original_label_counts = {}
label_counts = create_dictionary(Y_train_noaug)
for i in range(len(original_labels)):
    original_label_counts[i] = label_counts[original_labels[i]]

plt.figure(figsize=(18, 20))
plt.bar(*zip(*original_label_counts.items()))
plt.xticks(list(range(0, 9)),
           labels=[labels_dict[original_labels[i]][0:12] for i in range(len(original_labels))],
           rotation=60, fontsize='34')
fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
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