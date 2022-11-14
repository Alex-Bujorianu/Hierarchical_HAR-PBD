from Baseline.utils import MakeGraph
from data_extraction import loadalldata
import pandas as pd
import numpy as np
import os
from angleforAlex import get_half_skel_joint_angles

#print(MakeGraph())

data = pd.read_csv("EmoPainAtHome/rand_labels.csv")
print(len(data['Activity'].unique()))

datafolderpath = os.getcwd() + "/" + "EmoPainAtHome"
all_data = loadalldata(datafolderpath)

print("All data ", all_data)

rightjointids = {'chestbottom': "ChestBottom", 'thigh': "RightThigh", 'upperarm': "RightUpperArm",
                 'lowerleg': "RightLowerLeg", 'forearm': "RightForeArm", 'hip': "Hip"}
joints = ['chestbottom', 'thigh', 'upperarm', 'lowerleg', 'forearm', 'hip']
new_data = np.empty((0, 6, 3))

def count_session_length(arg: dict) -> int:
    for key, value in arg.items():
        for j in joints:
            if key == rightjointids[j]:
                return len(arg[key])

for dic in all_data.values():
    #print("Dic: ", dic.keys())
    for session in dic:
        #print("Inner dic keys: ", dic[session].keys())
        length = count_session_length(dic[session])
        print(length)
        #columns = dic[session][rightjointids[joints[0]]].reshape(-1, 1, 3)
        columns = np.empty((length, 1, 3))
        for j in range(0, len(joints)):
            try:
                bonedata = dic[session][rightjointids[joints[j]]]
                print(bonedata.shape)
                bonedata = bonedata.reshape(-1, 1, 3)
                print("Length of bonedata: ", len(bonedata))
                print("Shape of bonedata", bonedata.shape)
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
        print("Shape of columns ", columns.shape)
        if columns.shape[1] == new_data.shape[1]:
            # concatenate row-wise
            new_data = np.vstack((new_data, columns))
        else:
            continue
            # If a sensor is missing, skip this session

print(new_data)
print(new_data.shape)

print(new_data.flatten().shape)
# Need to window to convert 3D to 4D
angles = get_half_skel_joint_angles(new_data)
print(angles)


