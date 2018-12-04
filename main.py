import tensorflow as tf

import data_process as data
import model

from configs import DEFINES

def main(self):

    inputs, labels, t2i, i2t, max_len = data.load_data(DEFINES.data_path)
    encoder_inputs = inputs
    decoder_inputs, decoder_targets = prepare_dec(labels)

    embedding_matrix = data.get_embedding_matrix(DEFINES.data_path, DEFINES.embedding_path, i2t)

    params = make_params(embedding_matrix, max_len)
    estimator = tf.estimator.Estimator(model_fn = model.model_fn,
                                        model_dir = DEFINES.check_point,
                                        params = params)

    estimator.train(lambda:data.train_input_fn(encoder_inputs, decoder_inputs, decoder_targets))

def prepare_dec(labels):

    decoder_inputs = labels[:, :-1]
    decoder_targets = labels[:, 1:]

    return decoder_inputs, decoder_targets


def make_params(embedding_matrix, max_len):

    params = {'hidden_dim': DEFINES.hidden_dim,
            'embedding_matrix': embedding_matrix,
            'vocab_size': embedding_matrix.shape[0],
            'max_length': max_len,
            'teacher_forcing_rate': DEFINES.teacher_forcing_rate,
            'dropout_rate': DEFINES.dropout_rate,
            'learning_rate': DEFINES.learning_rate}

    return params

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
