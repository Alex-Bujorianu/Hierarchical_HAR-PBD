import os
import numpy
import tensorflow as tf
from tensorflow import keras
import sklearn
import copy


# angle data shape - (b, 240, 4, 3) - Your window size need to be 240 frames (6 seconds)
# pretrained_modelpath - file path for the hdf5 model I sent with this

def autofeats_extract(angledata, pretrained_modelpath):
    pretrainedmodel = keras.models.load_model(os.path.join(pretrained_modelpath),
                                              custom_objects={'weighted_mse': weighted_mse,
                                                              'MyRescale': MyRescale, 'GraphConvNN': GraphConvNN})

    encoder = keras.Model(pretrainedmodel.input, pretrainedmodel.get_layer(name='embedding_out').output)

    embeddings = numpy.zeros((angledata.shape[0], 20))

    activities = numpy.zeros((angledata.shape[0], 6))

    i_next = 0
    i_jump = 1000

    for i in numpy.arange(0, angledata.shape[0], i_jump):
        i_limit = min(i + i_jump, angledata.shape[0])

        embeddings[i:i_limit, :] = encoder.predict_on_batch(angledata[i:i_limit, :, :, :])

        activities[i:i_limit, :], _ = pretrainedmodel.predict_on_batch(angledata[i:i_limit, :, :, :])

        i_next += i_jump

    embeddings, _ = normalize(embeddings, scaletype='minmax', minmax_featurerange=(-1, 1))

    return embeddings, activities


def normalize(data, scaletype='maxabs', minmax_featurerange=(0, 1), transformer=None, subjectid=numpy.array([])):
    if transformer == None:
        newtransformers = True
        transformers = {}
    else:
        newtransformers = False
        transformers = copy.deepcopy(transformer)

    datashape = data.shape
    if len(datashape) == 3: data = numpy.reshape(data,
                                                 (datashape[0], -1))  # datashape --> (batchsize, sequencelen, dim)
    temp = copy.deepcopy(data)

    # print(subjectid)

    if subjectid.shape[0] != 0:

        # print("here")

        subjectset = {s for s in subjectid}

        for s in subjectset:

            if newtransformers:
                if scaletype.lower() == 'maxabs':
                    transformer = sklearn.preprocessing.MaxAbsScaler(copy=True)
                elif scaletype.lower() == 'minmax':
                    transformer = sklearn.preprocessing.MinMaxScaler(feature_range=minmax_featurerange, copy=True)
                else:
                    raise ValueError('This is not a recognised scale type')
            else:
                try:
                    transformer = transformers[s]
                except KeyError:
                    # print(s)
                    # print(transformers)
                    if scaletype.lower() == 'maxabs':
                        transformer = sklearn.preprocessing.MaxAbsScaler(copy=True)
                    elif scaletype.lower() == 'minmax':
                        transformer = sklearn.preprocessing.MinMaxScaler(feature_range=minmax_featurerange, copy=True)
                    else:
                        raise ValueError('This is not a recognised scale type')

            transformer.partial_fit(temp[subjectid == s, :])
            temp[subjectid == s, :] = transformer.transform(temp[subjectid == s, :])

            transformers[s] = copy.deepcopy(transformer)

        data = temp
        transformer = transformers

        # print(transformer)

    else:

        if newtransformers:
            if scaletype.lower() == 'maxabs':
                transformer = sklearn.preprocessing.MaxAbsScaler(copy=True)
            elif scaletype.lower() == 'minmax':
                transformer = sklearn.preprocessing.MinMaxScaler(feature_range=minmax_featurerange, copy=True)
            else:
                raise ValueError('This is not a recognised scale type')
        transformer.partial_fit(data)
        data = transformer.transform(data)

    if len(datashape) == 3: data = numpy.reshape(data, datashape)

    # maxbascaler min=-1, max =1
    return data, transformer


def weighted_mse(y_true, y_pred):
    gamma = 1
    beta = 2

    propn = tf.reduce_sum(y_true, axis=0, keepdims=True) / tf.reduce_sum(y_true)
    wgts = tf.reduce_sum(tf.math.pow(((1 - propn) + gamma), beta) * y_true, axis=1)

    # print(y_true)

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    base_mse = mse(y_true, y_pred)

    return tf.reduce_mean(base_mse * wgts)


class MyRescale(keras.layers.Layer):
    def __init__(self, randomseed, **kwargs):
        super(MyRescale, self).__init__(**kwargs)

        self.randomseed = tf.random.set_seed(randomseed)

    def call(self, inputs):
        scale = tf.random.uniform([tf.shape(inputs)[0], 1, 1, 1], minval=-2, maxval=2, seed=self.randomseed,
                                  dtype=tf.float32)
        scale = tf.repeat(scale, repeats=tf.shape(inputs)[1], axis=1)
        scale = tf.repeat(scale, repeats=tf.shape(inputs)[2], axis=2)
        scale = tf.repeat(scale, repeats=tf.shape(inputs)[3], axis=3)

        outputs = inputs * scale

        translation = tf.random.uniform([tf.shape(inputs)[0], 1, 1, 1], minval=-0.5, maxval=0.5, seed=self.randomseed,
                                        dtype=tf.float32)
        translation = tf.repeat(translation, repeats=tf.shape(inputs)[1], axis=1)
        translation = tf.repeat(translation, repeats=tf.shape(inputs)[2], axis=2)
        translation = tf.repeat(translation, repeats=tf.shape(inputs)[3], axis=3)

        outputs = inputs + translation

        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    ##    def compute_mask(self, inputs, mask=None):
    ##        # Also split the mask into 2 if it presents.
    ##        if mask is None:
    ##            return None
    ##        return mask

    def get_config(self):
        config = super(MyRescale, self).get_config()

        # Specify here all the values for the constructor's parameters
        config['randomseed'] = self.randomseed

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


### code borrowed from
# https://keras.io/examples/graph/gnn_citations/#build-a-graph-neural-network-model
# with minor edits


class GraphConvNN(keras.layers.Layer):
    def __init__(self, hidden_units, edges, edge_weights,
                 dropout_rate=0.0, aggregation_type="mean", combination_type="concat", normalize=False,
                 **kwargs):
        super(GraphConvNN, self).__init__(**kwargs)

        self.ffn = self.create_ffn(hidden_units, dropout_rate)

        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize
        self.edges = edges
        self.edge_weights = edge_weights

        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = self.create_ffn(hidden_units, dropout_rate)

    def create_ffn(self, hidden_units, dropout_rate, name=None):

        fnn_layers = []

        for units in hidden_units:
            # fnn_layers.append(keras.layers.BatchNormalization())
            # fnn_layers.append(keras.layers.Dropout(dropout_rate))
            fnn_layers.append(keras.layers.Dense(units, activation=tf.nn.tanh))

        return keras.Sequential(fnn_layers, name=name)

    def prepare(self, node_repesentations):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn(node_repesentations)

        return messages

    def aggregate(self, node_indices, neighbour_messages):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = tf.math.reduce_max(node_indices) + 1
        temp_neighbour_messages = tf.transpose(neighbour_messages, perm=[1, 0, 2])
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                temp_neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                temp_neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                temp_neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        aggregated_message = tf.transpose(aggregated_message, perm=[1, 0, 2])
        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=2)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=2)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")
        # print(h)
        # Apply the processing function.
        node_embeddings = self.update_fn(h)

        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=2)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        # print(node_embeddings)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        ###inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        # print(inputs)

        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = self.edges[0], self.edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_representations = tf.gather(inputs, neighbour_indices, axis=1)
        node_representations = tf.gather(inputs, node_indices, axis=1)

        # print(neighbour_representations)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_representations)
        # print(neighbour_messages)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(node_indices, neighbour_messages)
        # print(aggregated_messages)
        # Update the node embedding with the neighbour messages.

        out = self.update(inputs, aggregated_messages)

        # print(out)
        return out

    def compute_output_shape(self, input_shape):

        # print(input_shape)
        # print(int(self.hidden_units[len(self.hidden_units)-1]))
        return tf.TensorShape([None, int(input_shape[1]), int(self.hidden_units[len(self.hidden_units) - 1])])

    def get_config(self):
        config = super(GraphConvNN, self).get_config()

        # Specify here all the values for the constructor's parameters
        config['hidden_units'] = self.hidden_units
        config['dropout_rate'] = self.dropout_rate
        config['aggregation_type'] = self.aggregation_type
        config['combination_type'] = self.combination_type
        config['normalize'] = self.normalize
        config['edges'] = self.edges
        config['edge_weights'] = self.edge_weights

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


