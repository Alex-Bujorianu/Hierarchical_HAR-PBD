from using_pretrained_Alex import autofeats_extract
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from helper import merge_walking, pick_labels, new_encoding
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import OneHotEncoder

def take_argmax(activities_array):
    new_activities = np.empty(shape=(activities_array.shape[0]))
    for i in range(activities_array.shape[0]):
        new_activities[i] = np.argmax(activities_array[i, :])
    return new_activities

def plot_train_test_loss(predictor, X_train, Y_train, X_test, Y_test, N_EPOCHS=200, N_BATCH=200):
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_CLASSES = np.unique(Y_train)
    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            predictor.partial_fit(X_train[indices], Y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(predictor.score(X_train, Y_train))

        # SCORE TEST
        scores_test.append(predictor.score(X_test, Y_test))

        epoch += 1

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(scores_train)
    ax[0].set_title('Train')
    ax[1].plot(scores_test)
    ax[1].set_title('Test')
    fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.show()

def print_scores(Y_test, predictions):
    print("Accuracy: ", accuracy_score(Y_test, predictions))
    print("F1: ", f1_score(Y_test, predictions, average="weighted"))
    print("Precision: ", precision_score(Y_test, predictions, average='weighted'))
    print("Recall: ", recall_score(Y_test, predictions, average="weighted"))


X_train = np.load("Data/X_train_6s_40hz_pain.npy")
X_test = np.load("Data/X_test_6s_40hz_pain.npy")
Y_train = np.load("Data/Y_train_6s_40hz_pain.npy")
Y_test = np.load("Data/Y_test_6s_40hz_pain.npy")
new_encoding(Y_train)
new_encoding(Y_test)
# Merge walking should be called after new encoding
merge_walking(Y_train, Y_test)
to_predict = [12, 15, 1, 8, 2, 3, 14, 11, 10]
X_train, X_test, Y_train, Y_test = pick_labels(to_predict=to_predict, X_train=X_train, X_test=X_test,
                                               Y_train=Y_train, Y_test=Y_test)
# Y array 2D array → 1D array
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()
print("Classes in Y_train: ", np.unique(Y_train), "Classes in Y test: ", np.unique(Y_test))
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
embeddings_train, activities_train = autofeats_extract(X_train, "autoencoder.hdf5")
print("Shape of activities train ", activities_train.shape)
print("First 5 activities ", activities_train[0:5])
# I am guessing the activities are probabilities so select the most likely
new_activities_train = take_argmax(activities_train)
print(new_activities_train[0:5])
new_activities_train = new_activities_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
new_activities_train = encoder.fit_transform(new_activities_train)
print("New activities train ", new_activities_train[0:5])
print("Shape of new activities train: ", new_activities_train.shape)
print("Shape of embeddings train ", embeddings_train.shape)
# Add the columns
print(type(embeddings_train))
embeddings_train = np.hstack((embeddings_train, new_activities_train))
#print("Embeddings train: ", embeddings_train[0:10, :])
embeddings_test, activities_test = autofeats_extract(X_test, "autoencoder.hdf5")
new_activities_test = take_argmax(activities_test)
new_activities_test = new_activities_test.reshape(-1, 1)
new_activities_test = encoder.fit_transform(new_activities_test)
embeddings_test = np.hstack((embeddings_test, new_activities_test))
assert embeddings_test.shape[0] == X_test.shape[0]
print("Shape of embeddings train: ", embeddings_train.shape)
print("Shape of embeddings test: ", embeddings_test.shape)
print("Subset of embeddings_train ", embeddings_train[0:5],
      "Subset of embeddings_test", embeddings_test[0:5])
print("Y train (sanity check): ", Y_train[0:5])
print("Shapes of Y train and Y test: ", Y_train.shape, Y_test.shape)
best_params = json.load(open("optimisation_results.json"))
neural_network = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'], validation_fraction=0.2,
                               batch_size=best_params['batch_size'], solver='adam', early_stopping=True,
                               activation=best_params['activation'],
                               learning_rate_init=best_params['learning_rate_init'],
                               beta_1=best_params['beta_1'], beta_2=best_params['beta_2'],
                               epsilon=best_params['epsilon'], verbose=True, max_iter=200)
print(embeddings_train.shape)
#plot_train_test_loss(neural_network, X_train=embeddings_train, Y_train=Y_train, X_test=embeddings_test, Y_test=Y_test)
neural_network.fit(embeddings_train, Y_train)
plt.plot(neural_network.validation_scores_)
plt.title("Validation accuracy")
plt.show()
plt.plot(neural_network.loss_curve_)
plt.title("Loss during training")
plt.show()
print("Activation function in the output layer: ", neural_network.out_activation_)
print("Train predictions (sanity check): ", neural_network.predict(embeddings_train))
print("Y train: ", Y_train)
predictions = neural_network.predict(embeddings_test)
print("Predictions (test set): ", predictions)
print("Shape of predictions: ", predictions.shape)
print("Y test: ", Y_test)
# I noticed something odd
print_scores(Y_test, predictions)
conf_matrix = confusion_matrix(Y_test, predictions)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()

