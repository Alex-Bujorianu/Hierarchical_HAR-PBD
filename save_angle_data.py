from Baseline.utils import MakeGraph
from data_extraction import loadalldata, loadalllabels
import pandas as pd
import numpy as np
import os
import timeit
from angleforAlex import get_half_skel_joint_angles
import json
from helper import window

#print(MakeGraph())

data = pd.read_csv("EmoPainAtHome/rand_labels.csv")
activity_labels = data['Activity'].unique()
print("There are ", len(activity_labels), " labels")

datafolderpath = os.getcwd() + "/" + "EmoPainAtHome"
all_data = loadalldata(datafolderpath)

print(all_data['63'])
rightjointids = {'chestbottom': "ChestBottom", 'thigh': "RightThigh", 'upperarm': "RightUpperArm",
                 'lowerleg': "RightLowerLeg", 'forearm': "RightForeArm", 'hip': "Hip"}
joints = ['chestbottom', 'thigh', 'upperarm', 'lowerleg', 'forearm', 'hip']
# each session will contain sensor data and labels
new_data = {}

def count_session_length(arg: dict) -> int:
    for key, value in arg.items():
        for j in joints:
            if key == rightjointids[j]:
                return len(arg[key])

for key, dic in all_data.items():
    #print("key: ", key)
    one_session = np.empty((0, 6, 3))
    for session in dic:
        print("Session: ", session)
        length = count_session_length(dic[session])
        #print(length)
        #columns = dic[session][rightjointids[joints[0]]].reshape(-1, 1, 3)
        columns = np.empty((length, 1, 3))
        for j in range(0, len(joints)):
            try:
                bonedata = dic[session][rightjointids[joints[j]]]
                # print(bonedata.shape)
                bonedata = bonedata.reshape(-1, 1, 3)
                # print("Length of bonedata: ", len(bonedata))
                # print("Shape of bonedata", bonedata.shape)
                if len(bonedata) == 0:
                    break
                #concatenate column-wise
                columns = np.hstack((columns, bonedata))

            except KeyError:
                # many sessions only have 5 keys
                #previous_length = len(dic[session][rightjointids[joints[j-1]]])
                # empty_arr = np.empty((length, 1, 3))
                # empty_arr.fill(np.nan)
                # # Has to be the same length
                # columns = np.hstack((columns, empty_arr))
                break

        # Delete first column, no longer needed
        columns = np.delete(columns, 0, 1)
        # print("Shape of columns ", columns.shape)
        if columns.shape[1] == 6:
            new_data[key + "_" + session] = columns
        else:
            continue
            # If a sensor is missing, skip this session

print("New data has ", len(new_data.keys()), " keys")
#print("New data: ", new_data)
# Activity Labels
labelspath = os.getcwd() + "/" + "EmoPainAtHome/" + "rand_labels.csv"
Y = loadalllabels(labelspath)
#print("Y: ", Y)
json.dump(Y, open("label_data.json", "w"))
datetime = []
activity_labels = []
for key,value in Y.items():
    datetime.append(key)
    activity_labels.append(value['activity'])

print("Length of datetime ", len(datetime))
print("Length of activity labels ", len(activity_labels))


# Window everything
# This function is not slow
def windowify():
    for key,value in new_data.items():
        # print("Windowed array: ", window(value))
        print("Shape of windowed array: ", window(value).shape)
        new_data[key] = window(value)

windowify()

# This function is very slow
def convert_to_angles():
    for key,value in new_data.items():
        new_data[key] = get_half_skel_joint_angles(value)

convert_to_angles()
#print(new_data)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

json.dump(new_data, open("angle_data.json", "w"), cls=NumpyEncoder)

