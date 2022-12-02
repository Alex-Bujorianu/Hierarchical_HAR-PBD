# Instructions

The folder named EmoPain_old contains data about patients with chronic pain,
and is organised in a different way from the other two folders, EmoPainAtHomeFull
and EmoPainHealthy. The code to load them different: full_data_extraction
should be used instead of the other 2 scripts (save_angle_data and 
train_and_test). 

The folder named EmoPainHealthy contains only data about healthy patients.
It is organised in the same way as EmoPainAtHomeFull but is sampled at
10Hz instead of 40Hz.

The folder named Data contains the data, split, windowed and angled, as numpy arrays.
Healthy and pain refer to EmoPainHealthy and EmoPainAtHomeFull respectively.

The results folder contains the results of the different experiments,
including the confusion matrix as wel as accuracy, F1 score etc.

The Models folder contains saved Tensorflow models.