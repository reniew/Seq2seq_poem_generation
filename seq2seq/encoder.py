import tensorflow as tf

from tensorflow.keras.layers import Dropout

from .module import Bidirectional_gru

class Encoder:

    def __init__(self, hidden_dim, embedding_matrix, dropout_rate = 1.0):

        self.dropout = Dropout(dropout_rate)
        self.embedding_matrix = embedding_matrix
        self.bidirection = Bidirectional_gru(hidden_dim = hidden_dim)

    def build(self, inputs):

        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):

            embed_inputs = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
            embed_inputs = self.dropout(embed_inputs)
            encoder_outputs, encoder_state = self.bidirection.build(embed_inputs)

        return encoder_outputs, encoder_state
