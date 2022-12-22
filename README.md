# Instructions

The folder named EmoPain_old contains data about patients with chronic pain,
and is organised in a different way from the other two folders, EmoPainAtHomePain
and EmoPainAtHomeHealthy. The code to load them is different: full_data_extraction
should be used instead of the other 2 scripts (save_angle_data and 
train_and_test). 

The folder named EmoPainAtHomeHealthy contains only data about healthy patients.
It is organised in the same way as EmoPainAtHomePain but is sampled at
10Hz instead of 40Hz.

The folder named Data contains the data, split, windowed and angled, as numpy arrays.
Healthy and pain refer to EmoPainHealthy and EmoPainAtHomeFull respectively.

The results folder contains the results of the different experiments,
including the confusion matrix as well as accuracy, F1 score etc. They are JSON files.

The Models folder contains saved Tensorflow models.

Important note: the data is not contained in this repository for legal reasons. Please ask the data owner for permission to use the EmoPainAtHome dataset.

## MyHAR files

The my_HAR.ipynb and my_HAR.py both do the same thing. The Jupyter notebook exists only for use with Google Colab.
Use whichever is more convenient.

Bear in mind the following parameters:

* Timestep in the HAR_model_wrapper constructor refers to time*sampling rate. So a 3s window at 40Hz has a timestep of 120.
* If you decide to use joints instead of angles, you need a 6x6 adjacency matrix instead of a 4x4.


## Libraries

You need to have tensorflow (or tensorflow-gpu)
installed alongside numpy, pandas and matplotlib. The training time is considerable without a GPU (~45 min).