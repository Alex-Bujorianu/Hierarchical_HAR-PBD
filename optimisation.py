from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.space.space import Space
from skopt.utils import use_named_args
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np
from helper import merge_walking, pick_labels, new_encoding
from using_pretrained_Alex import autofeats_extract
from sklearn.metrics import accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from skopt.plots import plot_convergence

X_train = np.load("Data/X_train_6s_40hz_pain.npy")
X_test = np.load("Data/X_test_6s_40hz_pain.npy")
Y_train = np.load("Data/Y_train_6s_40hz_pain.npy")
Y_test = np.load("Data/Y_test_6s_40hz_pain.npy")
new_encoding(Y_train)
new_encoding(Y_test)
# Merge walking should be called after new encoding
merge_walking(Y_train, Y_test)
to_predict = [12, 15, 1, 8, 2, 3, 14, 11, 10]
X_train, X_test, Y_train, Y_test = pick_labels(to_predict=to_predict, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()
print("Shape of Y train: ", Y_train.shape)
embeddings_train, activities_train = autofeats_extract(X_train, "autoencoder.hdf5")
embeddings_test, activities_test = autofeats_extract(X_test, "autoencoder.hdf5")
print("Shape of embeddings train: ", embeddings_train.shape)

#hidden_layer_sizes =  Categorical(categories=[(20, 20), (200, 100, 20), (500, 200, 100)], name='hidden_layer_sizes')
learning_rate_init = Real(0.001, 0.005, name='learning_rate_init')
# Non-linear activation functions allows model to learn non-linear relationships
activation = Categorical(categories=['relu', 'logistic', 'identity'], name='activation')
space = dict()
space['activation'] = activation
#space['hidden_layer_sizes'] = hidden_layer_sizes
space['learning_rate_init'] = learning_rate_init
beta_1 = Real(0.8, 0.95, name='beta_1')
beta_2 = Real(0.99, 0.999, name='beta_2')
space['beta_1'] = beta_1
space['beta_2'] = beta_2
epsilon = Real(1e-8, 1e-1, name='epsilon')
space['epsilon'] = epsilon
estimator = MLPClassifier(batch_size=150, solver='adam', hidden_layer_sizes=(500, 200, 100), early_stopping=True)
search = BayesSearchCV(estimator, space, scoring='f1_macro', n_jobs=4)
search.fit(embeddings_train, Y_train)
print("Best params: ", search.best_params_)
print("Best score: ", search.best_score_)