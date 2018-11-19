import numpy as np
import tensorflow as tf



def LSTM_Network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs], name="features_reshape")
    hidden = tf.nn.relu(tf.matmul(
        feature_mat, config.W['hidden']
    ) + config.biases['hidden'])
    hidden = tf.split(0, config.n_steps, hidden, name="input_hidden")
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    outputs, _ = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)
    lstm_last_output = outputs[-1]
    # Linear activation
    final_out = tf.add(tf.matmul(lstm_last_output, config.W['output']), config.biases['output'], name="logits")
    
    return final_out
