from Baseline.HierarchicalHAR_PBD import build_model
import numpy as np
import Baseline.utils as utils

adjacency_matrix = np.transpose(np.array([[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]])).reshape((4, 4))
print("Adjacency matrix: ", adjacency_matrix)
utils.MakeGraph(adjacency_matrix)

class HAR_model_wrapper():
    "A class to hold a HAR model and its important properties"
    timestep = 0
    node_num = 0
    feature_num = 0
    def __init__(self, adjacency_matrix, timestep, node_num, feature_num, num_class_HAR=26):
        self.model = build_model(timestep=timestep, body_num=node_num, feature_dim=feature_num,
                              gcn_units_HAR=26, lstm_units_HAR=24, adjacency_matrix=adjacency_matrix,
                              gcn_units_PBD=16, lstm_units_PBD=24,
                              num_class_HAR=num_class_HAR, num_class_PBD=2)[1]
        self.timestep = timestep
        self.node_num = node_num
        self.feature_num = feature_num



HARmodel = HAR_model_wrapper(adjacency_matrix=adjacency_matrix,
                             timestep=120, node_num=4, feature_num=3)


def train_model(model: HAR_model_wrapper, X_train: np.ndarray, Y_train: np.ndarray):
    AdjNorm = utils.MakeGraph(model.node_num)
    graphtrain = utils.combine(AdjNorm, X_train, model.timestep, model.node_num, model.feature_num)
    model.model.compile(optimizer=Adam(lr=5e-4, decay=1e-5),
                  loss={'PBDout': utils.focal_loss(weights=utils.class_balance_weights(0.9999,
                                                                                       [np.sum(y_train_PBD[:, 0]),
                                                                                        np.sum(y_train_PBD[:, 1])]),
                                                   gamma=2, num_class=2),
                        'HARout': 'categorical_crossentropy'
                        },
                  loss_weights={'PBDout': 1., 'HARout': 1.},
                  metrics=['categorical_accuracy'])

    model.model.fit(x=graphtrain,
              y=Y_train,
              batch_size=150,
              epochs=100,
              callbacks=utils.build_callbacks('Model', str(valid_patient)),
              validation_data=(graphvalid, (y_valid_HAR, y_valid_PBD))
              )