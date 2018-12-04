import tensorflow as tf

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 1, 'epoch')
tf.app.flags.DEFINE_integer('hidden_dim', 128, 'hidden dim')
tf.app.flags.DEFINE_integer('embedding_dim', 128, 'embedding dim')
tf.app.flags.DEFINE_float('teacher_forcing_rate', 0.5, 'teacher forcing rate')
tf.app.flags.DEFINE_float('dropout_rate', 0.5, 'dropout rate')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('data_path', './data/poem_data.json', 'data path')
tf.app.flags.DEFINE_string('embedding_path', './data/embedding_matrix.npy', 'embeding path')
tf.app.flags.DEFINE_string('check_point', './check_point', 'chech_point')

DEFINES = tf.app.flags.FLAGS
