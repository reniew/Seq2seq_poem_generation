import tensorflow as tf

from tensorflow.keras.layers import Dropout, GRU, Dense

from .module import Additive_attention


class Decoder:

    def __init__(self,
                max_length,
                is_train,
                teacher_forcing_rate,
                hidden_dim,
                embedding_matrix,
                dropout_rate = 1.0):

        self.max_length = max_length
        self.teacher_forcing_rate = teacher_forcing_rate
        self.is_train = is_train
        self.dropout = Dropout(dropout_rate)
        self.embedding_matrix = embedding_matrix
        self.attention = Additive_attention(num_units = hidden_dim)
        self.gru = GRU(units = hidden_dim, return_state = True)
        self.dense = Dense(units = embedding_matrix.shape[0])
        # GRU 하나만 구현 성능 낮으면 Helper와 dynamic decoder 이용필요

    def build(self, encoder_outputs, encoder_state, decoder_inputs):

        with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):

            prediction = []
            logits = []
            decoder_state = encoder_state
            for i in range(self.max_length + 1):
                with tf.variable_scope('{}th_block'.format(i+1), reuse = tf.AUTO_REUSE):

                    if i > 0:
                        input_embed = self._teacher_forcing(output_token, decoder_inputs, i)
                    else:
                        input_embed = tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs[:, 0])

                    input_embed = self.dropout(input_embed)

                    decoder_state = self.attention.build(encoder_outputs, decoder_state)
                    decoder_outputs, decoder_state = self.gru(inputs = tf.expand_dims(input_embed,1),
                                                            initial_state = decoder_state)

                    output_logit = self.dense(decoder_outputs)
                    output_token = tf.argmax(output_logit, axis = -1)

                    logits.append(output_logit)
                    prediction.append(output_token)

            prediction, logits = self._reshape_pred_logit(prediction, logits)

        return prediction, logits


    def _teacher_forcing(self, output_token, decoder_inputs, idx):

        random_value = tf.random_uniform(shape = (), maxval = 1)
        input_embed = tf.cond(tf.logical_and(self.is_train, (random_value <= self.teacher_forcing_rate)),
                                lambda: tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs[:, idx]),
                                lambda: tf.nn.embedding_lookup(self.embedding_matrix, output_token))

        return input_embed

    def _reshape_pred_logit(self, prediction, logits):
        prediction = tf.transpose(tf.stack(prediction))
        logits = tf.transpose(tf.stack(logits), perm=[1,0,2])

        return prediction, logits
