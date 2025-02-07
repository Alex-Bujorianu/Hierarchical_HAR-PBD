import unittest
from helper import get_all_data, remove_empty_string, \
    rebalance_classes, window, convert_windowed_array_to_shape, convert_windowed_Y_to_shape
import csv
import numpy as np

class TestDataExtraction(unittest.TestCase):
    def test_get_all_data(self):
        X, Y = get_all_data("test_data")
        # print(Y[0:10])
        # print(X[0:10])
        bones_we_need = [
            "ChestBottom",
            "RightThigh",
            "RightUpperArm",
            "RightLowerLeg",
            "RightForeArm",
            "Hip"
        ]
        i = 0
        bone = "ChestBottom"
        reader = csv.reader(open("test_data/P213_Filing documents/Positions_" + bone + ".csv"))
        next(reader)
        arr = list(reader)
        remove_empty_string(arr)
        float_arr = [[float(y) for y in x] for x in arr]
        column = np.array(float_arr, dtype=object)
        print("X[0, 0, 0]", X[0, 0, i].all())
        print(column)
        self.assertEqual(column[0, 2], X[0, 0, i].all())
        self.assertEqual(list(column[0]), [-0.001, 0.396, 0.026])
        self.assertEqual(list(X[0, 0, i]), [-0.001, 0.396, 0.026])
        self.assertEqual(Y[i], 6)
        bone = "Hip"
        reader = csv.reader(open("test_data/P213_Filing documents/Positions_" + bone + ".csv"))
        next(reader)
        arr = list(reader)
        remove_empty_string(arr)
        float_arr = [[float(y) for y in x] for x in arr]
        column = np.array(float_arr, dtype=object)
        i = 5
        self.assertEqual(list(column[0]), [0.002, 0.039, -0.008])
        self.assertEqual(list(X[0, 0, i, :]), [0.002, 0.039, -0.008])
    def test_rebalance_classes(self):
        X, Y = get_all_data("test_data")
        X_train, Y_train, X_test, Y_test = rebalance_classes(X, Y, split_ratio=0.6)
        self.assertEqual(Y_train[0], 6)
        self.assertEqual(Y_test[0], 6)



if __name__ == '__main__':
    unittest.main()
