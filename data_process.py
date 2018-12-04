import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import json
import os
import re

from gensim.models import Word2Vec

from configs import DEFINES


def get_embedding_matrix(data_path, embedding_path, i2t):
    if os.path.isfile(embedding_path):
        return np.load(open(embedding_path, 'rb')).astype(np.float32)
    else:
        make_embedding(data_path, embedding_path, i2t)
        return np.load(open(embedding_path, 'rb')).astype(np.float32)


def make_embedding(data_path, embedding_path, i2t):

    all_sent = []
    token_sent = []
    filter = re.compile("([~.,!?\"':;)(<=>&%$#@_\]\[\+\-\*\^])")

    data = json.load(open(data_path, 'r', encoding = 'utf-8'))

    for v in data.values():
        all_sent.extend(v)

    for v in all_sent:
        token_sent.append(re.sub(filter,"",v).split())

    model = Word2Vec(token_sent,
                    size = DEFINES.embedding_dim,
                    window=5,
                    min_count=1,
                    workers=4,
                    sg=1,
                    iter = 10,
                    sample = 1e-3)

    embedding_matrix = special_token_embedding(model)
    unk = embedding_matrix[3]

    for i, t in i2t.items():
        if i>3:
            if t in model.wv.vocab:
                embedding_matrix.append(model.wv.word_vec(t))
            else:
                embedding_matrix.append(unk)

    np.save(open(embedding_path, 'wb'), np.stack(embedding_matrix))


def special_token_embedding(model):
    pad = np.zeros(shape = (DEFINES.embedding_dim), dtype=np.float32)
    start = np.random.uniform(low=model.wv.vectors.min(), high=model.wv.vectors.max(), size=(DEFINES.embedding_dim))
    end = np.random.uniform(low=model.wv.vectors.min(), high=model.wv.vectors.max(), size=(DEFINES.embedding_dim))
    unk = np.random.uniform(low=model.wv.vectors.min(), high=model.wv.vectors.max(), size=(DEFINES.embedding_dim))

    return [pad, start, end, unk]







def make_vocab(vocab, pad, start, end, unk):

    t2i = {'<PAD>': pad, '<BOS>': start, '<EOS>': end, '<UNK>': unk}
    i2t = {pad: '<PAD>', start: '<BOS>', end: '<EOS>', unk: '<UNK>'}

    for word, idx in vocab.items():
        t2i[word] = idx + 3
        i2t[idx + 3] = word

    return t2i, i2t

def load_data(file_path):
    '''
    Load numpy data

    Args:
        inputs: json data file path, key => title, value => poem contents

    Return:
        inputs:
        labels:
        t2i:
        i2t:
    '''

    pad_token = 0 # <PAD>
    start_token = 1 # <BOS>
    end_token = 2 # <EOS>
    unk_token = 3 #<UNK>


    data = json.load(open(file_path, 'r', encoding='utf-8'))
    data_list = []
    all_sentences = []

    for poem in data.values():
        data_list.append(poem)
        all_sentences.extend(poem)

    source = []
    target = []

    for item in data_list:
        source.extend(item[:-1])
        target.extend(item[1:])

    max_len = int(round(np.percentile(np.array([len(x.split(' ')) for x in all_sentences]), 99)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)
    source = tokenizer.texts_to_sequences(source)
    target = tokenizer.texts_to_sequences(target)

    assert len(source) == len(target)

    for i in range(len(source)):
        source[i] = np.add(source[i], 3)
        target[i] = np.hstack(([start_token], np.add(target[i], 3), [end_token]))

    inputs = pad_sequences(source, maxlen=max_len, padding = 'post')
    labels = pad_sequences(target, maxlen=max_len+2, padding = 'post')


    t2i, i2t = make_vocab(tokenizer.word_index, pad_token, start_token, end_token, unk_token)

    return inputs, labels, t2i, i2t, max_len

def mapping_fn(enc_input, dec_input, dec_target):
    features = {"encoer_inputs": enc_input, "decoder_inputs": dec_input}
    labels = dec_target
    return features, labels


def train_input_fn(encoder_inputs, decoder_inputs, decoder_targets):
    dataset = tf.data.Dataset.from_tensor_slices((encoder_inputs, decoder_inputs, decoder_targets))
    dataset = dataset.shuffle(len(encoder_inputs))
    dataset = dataset.batch(DEFINES.batch_size)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(DEFINES.epoch)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

def token2str(token_data, i2t):
    output = []

    for idx in token_data:
        if idx > 2:
            output.append(i2t[idx])

    return ' '.join(output)

def str2token(str_data, t2i, max_len):
    output = []
    data = str_data.split()

    for token in data:
        output.append(t2i[token])

    pad = [0]*(max_len-len(output))

    return np.array(output+pad)
