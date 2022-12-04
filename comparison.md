## Matching

+ If Y_t is bigger than X_t but smaller than X_t+1, return X_t, even if Y_t is closer to X_t+1.
+ For loop skips keys that are too early. It increments 60s in time until it finds one that is later.
+ Performance: 10038 samples in train, 3880 samples in test.
+ Unique activities: [ 1.  2.  3.  6. 11. 14. 17. 20. 25.] in train and [ 2.  3.  5.  7.  9. 11. 16. 20. 21. 23. 24. 26.] in test = 4 labels in common.

New implementation:

+ Distance is calculated and the closer key is returned.
+ Performance: 10158 samples in train, 3880 samples in test.
+ A marginal improvement

## Merging labels

Baseline, with no merging:

1. F1 score: 0.445
2. Accuracy: 0.496
3. Precision: 0.486
4. Recall: 0.496

* Option 1: merge 9, 11 and 20 (washing machine activities) and painting shelves & painting wall.
* Option 2: Like option 1, but also merge 21 and 22 (vacuuming and vacuuming car)

Option 1 yields a significant improvement in accuracy and F1:

1. F1 score  0.5810524479223328
2. Accuracy  0.5980392156862745
3. Precision  0.5914676389789272
4. Recall  0.5980392156862745

Option 2:

1. F1 score  0.45298212168395685
2. Accuracy  0.5003267973856209
3. Precision  0.49458705216285054
4. Recall  0.5003267973856209

Option 2 is a lot worse than option 1.

## Increasing time window
Increasing the time window gives the model more data to recognise an activity, but, it effectively reduces the number of training instances. I think the learning rate should be larger to account for that.
