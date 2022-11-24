import numpy as np
import pandas as pd
# Labels 9, 11 and 20 are basically the same thing
# 21 and 22 (vacuuming and vacuuming car) are also similar
# Painting shelves and painting wall?
def merge_option_1(Y: np.ndarray) -> np.ndarray:
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