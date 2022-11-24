import json
from json import JSONDecodeError
import os
import numpy as np
import csv
import pandas as pd
from helper import create_mapping

labels_csv = pd.read_csv("EmoPainAtHomeFull/labels.csv")

def remove_empty_string(input: list) -> list:
    #@param input: a list of lists
    #@return: a list of lists with the empty string at the end of each sublist removed
    #Also removes first element
    for i in range(len(input)):
        input[i] = input[i][1:-1]

def append_value_multiple_times(arr: list, label: int, n_times: int):
    for i in range(n_times):
        arr.append(label)

def get_all_data(folderpath: str) -> (np.ndarray, np.ndarray):
    labels = create_mapping(labels_csv)
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
                    if activity_name=='Loading and unloading washing machine':
                        activity_name = 'Loading & unloading washing machine'
                    try:
                        activity_num = labels[activity_name]
                    except KeyError:
                        print("Unknown activity encountered, skipping…")
                        continue
                    columns = np.empty(shape=(0, 3), dtype=object)
                    for bone in bones_we_need:
                        reader = csv.reader(open(it.path + "/" + "Positions_" +  bone + ".csv"))
                        next(reader)
                        arr = list(reader)
                        remove_empty_string(arr)
                        arr = np.array(arr, dtype=object)
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
    Y = np.array(Y, dtype=object)
    return (X, Y)

X, Y = get_all_data("EmoPainAtHomeFull")
print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)