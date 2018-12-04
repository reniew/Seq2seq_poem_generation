import tensorflow as tf
import os
import sys

import model
import data_process as data


from configs import DEFINES

def main(self):

    arg_length = len(sys.argv)

    if(arg_length < 2):
        raise Exception("You should put one sentences to predict")

    inputs = []
    for i in sys.argv[1:]:
        inputs.append(i)

    inputs = " ".join(inputs)

    _, _, t2i, i2t, max_len = data.load_data(DEFINES.data_path)

    encoder_inputs, decoder_inputs = prepare_pred_input(inputs)

    embedding_matrix = data.get_embedding_matrix(DEFINES.data_path, DEFINES.embedding_path, i2t)

    params = make_params(embedding_matrix, max_len)

    estimator = tf.estimator.Estimator(model_fn = model.model_fn,
                                        model_dir = DEFINES.check_point,
                                        params = params)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"encoer_inputs": encoder_inputs, "decoer_inputs": decoder_inputs},
        num_epochs=1,
        shuffle=False)
    predict = estimator.predict(input_fn = predict_input_fn)

    prediction = next(predict)['prediction']

    return data.token2str(prediction, i2t)

def prepare_pred_input(inputs, t2i):

    result = []
    for token in inputs.split():
        if token in t2i:
            result.append(t2i[token])
        else:
            result.append(t2i['<UNK>'])

    encoder_inputs = np.array(result)
    decoder_inputs = np.array([1])

    return encoder_inputs, decoder_inputs


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
