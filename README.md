# Text Generation Model by Sequence to Seqeunce with Attention

Attention based sequence to sequence model with tensorflow

Koeran Poem generation model trained by korean poem dataset

# Dependencies

* Python >= 3.6
* Tensorflow >= 1.8
* gensim
* numpy

# Datasets

Crawled koeran poem dataset

* Total 1034 poem
* Total 17066 sequnece to trainable

# Config

```python
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
```

# Usage

Install requirements

```
pip install -r requirements.txt
```

Training model

```
python main.py
```


# Predict

For prediction, you should put one sentence used by input

**example**

```
python predict.py 맨발로 거친 산길을 오르는 나의 발바닥은 돌멩이에 찢겨 나뭇가지에 찢겨
```

# Generated Poem Example

```
Input: 맨발로 거친 산길을 오르는 나의 발바닥은 돌멩이에 찢겨 나뭇가지에 찢겨

먹다버린 깨진 콜라병과 눈총과 온갖 쓰레기에 치여
검붉은 피로 멍들어
불혹의 동반자로 떠도는 편지를
그 사이로 일은
곱게 물째 은행잎을 바라보지
```


# Reference Paper

* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
