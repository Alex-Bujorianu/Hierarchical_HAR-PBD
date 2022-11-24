import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import json
from sklearn.metrics import ConfusionMatrixDisplay
Y_train = np.load("Data/Y_train.npy")
Y_test = np.load("Data/Y_test.npy")

print("Shape of Y_train: ", Y_train.shape)

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

plt.bar(*zip(*train_labels.items()))
plt.title("Label distribution of training set")
plt.xticks(list(train_labels.keys()))
plt.show()
plt.bar(*zip(*test_labels.items()))
plt.title("Label distribution of test set")
plt.xticks(list(test_labels.keys()))
fig = plt.gcf()
fig.set_size_inches(12.0, 8)
plt.show()
labels_csv = pd.read_csv("EmoPainAtHome/rand_labels.csv")
def create_mapping(data: pd.DataFrame) -> dict:
    mappings = {}
    labels = data['Activity'].to_list()
    codes = data['Activity_recoded'].to_list()
    for i in range(len(labels)):
        mappings[labels[i]] = codes[i]
    return mappings

mappings = create_mapping(labels_csv)
print(mappings)

conf_matrix_cfcc = np.array(json.load(open("Results/Experiment_CFCC_repro_26l", "r"))['Confusion matrix'])
print("Length of conf matrix cfcc ", len(conf_matrix_cfcc[0]))
conf_matrix_catloss = np.array(json.load(open("Results/Experiment_cat_loss_reproducible", "r"))['Confusion matrix'])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_cfcc, display_labels=np.array(list(range(1, 27))))
fig, ax = plt.subplots(figsize=(16, 16))
plt.rcParams.update({'font.size': 16})
disp.plot(ax=ax)
plt.show()
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_catloss)
disp.plot()
plt.show()
