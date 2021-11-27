# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


from collections import Counter
import itertools
import numpy as np
import pandas as pd
import h5py
import six
import pickle
import os
import sklearn.preprocessing as sklearn_preprocess


class Tokenizer(object):
    def __init__(self, topk_words=None, na_value=None, min_freq=1, splitter=None, 
                 lower=False, oov_token=0, max_len=0, padding="pre"):
        self._topk_words = topk_words
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.oov_token = oov_token # use 0 for __OOV__
        self.word_counts = Counter()
        self.vocab = dict()
        self.vocab_size = 0 # include oov and padding
        self.max_len = max_len
        self.padding = padding

    def fit_on_texts(self, texts, use_padding=True):
        tokens = list(texts)
        if self._splitter is not None: # for sequence
            text_splits = [text.split(self._splitter) for text in texts if not pd.isnull(text)]
            if self.max_len == 0:
                self.max_len = max(len(x) for x in text_splits)
            tokens = list(itertools.chain(*text_splits))
        if self._lower:
            tokens = [tk.lower() for tk in tokens]
        if self._na_value is not None:
            tokens = [tk for tk in tokens if tk != self._na_value]
        self.word_counts = Counter(tokens)
        words = [token for token, count in self.word_counts.most_common() if count >= self._min_freq]
        self.word_counts.clear() # empty the dict to save memory
        if self._topk_words:
            words = words[0:self._topk_words]
        self.vocab = dict((token, idx) for idx, token in enumerate(words, 1 + self.oov_token))
        self.vocab["__OOV__"] = self.oov_token
        if use_padding:
            self.vocab["__PAD__"] = len(words) + self.oov_token + 1 # use the last index for __PAD__
        self.vocab_size = len(self.vocab) + self.oov_token

    def encode_category(self, categories):
        category_indices = [self.vocab.get(x, self.oov_token) for x in categories]
        return np.array(category_indices)

    def encode_sequence(self, texts):
        sequence_list = []
        for text in texts:
            if pd.isnull(text) or text == '':
                sequence_list.append([])
            else:
                sequence_list.append([self.vocab.get(x, self.oov_token) for x in text.split(self._splitter)])
        sequence_list = padding(sequence_list, maxlen=self.max_len, value=self.vocab_size - 1,
                                padding=self.padding, truncating=self.padding)
        return np.array(sequence_list)
    
    def load_pretrained_embedding(self, feature_name, pretrain_path, embedding_dim, output_path):
        with h5py.File(pretrain_path, 'r') as hf:
            keys = hf["key"][:]
            pretrained_vocab = dict(zip(keys, range(len(keys))))
            pretrained_emb = hf["value"][:]
        embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(self.vocab_size, embedding_dim))
        for word, idx in self.vocab.items():
            if word in pretrained_vocab:
                embedding_matrix[idx] = pretrained_emb[pretrained_vocab[word]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'a') as hf:
            hf.create_dataset(feature_name, data=embedding_matrix)

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab) + self.oov_token
            
        
class Normalizer(object):
    def __init__(self, normalizer):
        if not callable(normalizer):
            self.callable = False
            if normalizer in ['StandardScaler', 'MinMaxScaler']:
                self.normalizer = getattr(sklearn_preprocess, normalizer)()
            else:
                raise NotImplementedError('normalizer={}'.format(normalizer))
        else:
            # normalizer is a method
            self.normalizer = normalizer
            self.callable = True

    def fit(self, X):
        if not self.callable:
            null_index = np.isnan(X)
            self.normalizer.fit(X[~null_index].reshape(-1, 1))

    def normalize(self, X):
        if self.callable:
            return self.normalizer(X)
        else:
            return self.normalizer.transform(X.reshape(-1, 1)).flatten()


def padding(sequences, maxlen=None, dtype='int32',
            padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length """
    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)
    
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr

