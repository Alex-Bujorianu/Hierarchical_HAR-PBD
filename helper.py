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
    print((len(arr)-1,) + arr[0].shape)
    arr_to_return = np.empty(shape=(len(arr)-1,) + arr[0].shape)
    # Do not use vstack, numpy arrays are very slow to append
    # Pre-allocate memory in one big array and index
    for i in range(len(arr_to_return)):
        arr_to_return[i] = arr[i]
    return arr_to_return