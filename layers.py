from GCN_TF2.inits import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from GCN_TF2.config import args

from Helper_Functions import *

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, rate):
    """
    Dropout for sparse tensors.
    """
    noise_shape = [len(x.values)]
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Dense(layers.Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    """

    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 sampling=False,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.sampling = sampling

        self.weights_ = []
        for i in range(1):
            w = self.add_weight(name='weight' + str(i), shape=[input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight(name='bias', shape=[output_dim, 1])

        # for p in self.trainable_variables:
        #     print(p.name, p.shape)

    def call(self, inputs, in_sample=None, dist=None, out_sample=None, training=None):
        x, support_ = inputs
        if self.sampling:
            N = int(support_[0].dense_shape[0])
            if out_sample is None:
                out_sample = [i for i in range(N)]

            num_samples = len(dist)
            dist = np.expand_dims(dist, 1)

            for idx, mat in enumerate(support_):
                M = tf.sparse.to_dense(tf.sparse.reorder(mat)).numpy()
                sample_adj = M[out_sample]
                sample_adj = sample_adj[:, in_sample]

                sample_adj /= dist * 1000

                sample_adj /= num_samples / 1000
                # sample_adj *= len(dist)/num_in_sample
                sample_adj = tf.convert_to_tensor(sample_adj)
                sample_adj = tf.cast(sample_adj, dtype=tf.float32)
                support_[idx] = tf.sparse.from_dense(sample_adj)

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)

        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless:  # if it has features x
                pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]

            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.bias

        return self.activation(output)
