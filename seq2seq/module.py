import tensorflow as tf

from tensorflow.keras.layers import Dense, GRU


class Additive_attention:

    def __init__(self, num_units, reuse = tf.AUTO_REUSE):
        self.W1 = Dense(num_units)
        self.W2 = Dense(num_units)
        self.V = Dense(1)
        self.dense = Dense(num_units)
        self.reuse = reuse


    def build(self, encoder_outputs, decoder_state):
        with tf.variable_scope('additive_attention', self.reuse):
            decoder_state_3dim = tf.expand_dims(decoder_state, 1)

            score =  self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_state_3dim)))

            attention_weights = tf.nn.softmax(score, axis = 1)

            context_vector_temp = attention_weights * encoder_outputs
            context_vector = tf.reduce_sum(context_vector_temp, axis = 1)
            context_vector = self.dense(context_vector)

            return context_vector


class Bidirectional_gru:

    def __init__(self,
                hidden_dim,
                initializer = 'glorot_uniform',
                return_sequences = True,
                return_state = True,
                reuse = tf.AUTO_REUSE):

        self.reuse = reuse
        self.forward_gru = GRU(units = hidden_dim,
                               recurrent_initializer=initializer,
                               recurrent_activation='sigmoid',
                               return_sequences=return_sequences,
                               return_state=return_state,
                               dtype=tf.float32)
        self.backward_gru = GRU(units = hidden_dim,
                               recurrent_initializer=initializer,
                               recurrent_activation='sigmoid',
                               return_sequences=return_sequences,
                               return_state=return_state,
                               dtype=tf.float32)
        self.dense = Dense(units = hidden_dim)


    def build(self, inputs):

        with tf.variable_scope('bidirectional_GRU', self.reuse):
            forward_outputs, forward_state = self.forward_gru(inputs)
            backward_outputs, backward_state = self.backward_gru(tf.reverse(inputs, axis = [-1]))
            hidden_state = self.dense(tf.concat((forward_state, backward_state), axis = -1))
        return tf.concat((forward_outputs, backward_outputs), axis = -1), hidden_state
