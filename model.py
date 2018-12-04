import tensorflow as tf

import seq2seq


def model_fn(mode, features, labels, params):

    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    graph = seq2seq.Graph(mode, params)
    predict, logits = graph.build(features['encoer_inputs'], features['decoder_inputs'])

    if PREDICT:
        predictions = {'prediction': predict}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = make_loss(logits, labels)

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())


    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)


def make_loss(logits, labels):

    with tf.variable_scope('loss'):
        target_lengths = tf.reduce_sum(
                tf.to_int32(tf.not_equal(labels, 0)), 1)

        weight_masks = tf.sequence_mask(
                    lengths=target_lengths,
                    maxlen=labels.shape.as_list()[1],
                    dtype=tf.float32, name='masks')

        loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=labels,
                weights=weight_masks,
                name="squence_loss")
    return loss
