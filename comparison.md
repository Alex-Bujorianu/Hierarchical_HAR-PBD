Current implementation:

+ If Y_t is bigger than X_t but smaller than X_t+1, return X_t, even if Y_t is closer to X_t+1.
+ For loop skips keys that are too early. It increments 60s in time until it finds one that is later.
+ Performance: 10038 samples in train, 3880 samples in test. 
+ Unique activities: [ 1.  2.  3.  6. 11. 14. 17. 20. 25.] in train and [ 2.  3.  5.  7.  9. 11. 16. 20. 21. 23. 24. 26.] in test = 4 labels in common. 

New implementation:

+ Distance is calculated and the closer key is returned.
+ Performance: 10158 samples in train, 3880 samples in test.
+ A marginal improvement
