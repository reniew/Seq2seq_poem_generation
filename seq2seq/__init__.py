import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder


class Graph:

    def __init__(self, mode, params):

        self.mode = mode
        self.hidden_dim = params['hidden_dim']
        self.embedding_matrix = params['embedding_matrix']
        self.max_length = params['max_length']
        self.teacher_forcing_rate = params['teacher_forcing_rate']
        self.dropout_rate = params['dropout_rate']
        self.is_train = (mode == tf.estimator.ModeKeys.TRAIN)


    def build(self, encoder_inputs, decoder_inputs):

        encoder_inputs = tf.nn.embedding_lookup(params = self.embedding_matrix,
                                                ids = self.encoder_inputs)

        decoder_inputs = tf.nn.embedding_lookup(params = self.embedding_matrix,
                                                ids = self.decoder_inputs)
        
        encoder = Encoder(hidden_dim = self.hidden_dim,
                        embedding_matrix = self.embedding_matrix,
                        dropout_rate = self.dropout_rate)

        decoder = Decoder(max_length = self.max_length,
                        is_train = self.is_train,
                        teacher_forcing_rate = self.teacher_forcing_rate,
                        hidden_dim = self.hidden_dim,
                        embedding_matrix = self.embedding_matrix,
                        dropout_rate = self.dropout_rate)

        encoder_outputs, encoder_state = encoder.build(encoder_inputs)
        prediction, logits = decoder.build(encoder_outputs, encoder_state, decoder_inputs)

        return prediction, logits
