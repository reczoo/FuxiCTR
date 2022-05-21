# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from collections import Counter
import itertools
import numpy as np
import pandas as pd
import h5py
import pickle
import json
import os
from tqdm import tqdm
import sklearn.preprocessing as sklearn_preprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


class Tokenizer(object):
    def __init__(self, num_words=None, na_value=None, min_freq=1, splitter=None, 
                 lower=False, oov_token=0, max_len=0, padding="pre", num_workers=4):
        self._num_words = num_words
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.oov_token = oov_token # use 0 for __OOV__
        self.vocab = dict()
        self.vocab_size = 0 # include oov and padding
        self.max_len = max_len
        self.padding = padding
        self.num_workers = num_workers
        self.use_padding = False

    def fit_on_texts(self, texts, use_padding=False):
        self.use_padding = use_padding
        word_counts = Counter()
        if self._splitter is not None: # for sequence
            max_len = 0
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                chunks = np.array_split(texts, self.num_workers)
                tasks = [executor.submit(count_tokens, chunk, self._splitter) for chunk in chunks]
                for future in tqdm(as_completed(tasks), total=len(tasks)):
                    block_word_counts, block_max_len = future.result()
                    word_counts.update(block_word_counts)
                    max_len = max(max_len, block_max_len)
            if self.max_len == 0:  # if argument max_len not given
                self.max_len = max_len
        else:
            word_counts = Counter(list(texts))
        self.build_vocab(word_counts)

    def build_vocab(self, word_counts):
        # sort to guarantee the determinism of index order
        word_counts = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        words = []
        for token, count in word_counts:
            if count >= self._min_freq:
                if self._na_value is None or token != self._na_value:
                    words.append(token.lower() if self._lower else token)
        if self._num_words:
            words = words[0:self._num_words]
        self.vocab = dict((token, idx) for idx, token in enumerate(words, 1 + self.oov_token))
        self.vocab["__OOV__"] = self.oov_token
        if self.use_padding:
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
        sequence_list = pad_sequences(sequence_list, maxlen=self.max_len, value=self.vocab_size - 1,
                                      padding=self.padding, truncating=self.padding)
        return np.array(sequence_list)
    
    def load_pretrained_embedding(self, feature_name, pretrain_path, embedding_dim, 
                                  output_path, feature_dtype=str, freeze_emb=True):
        with h5py.File(pretrain_path, 'r') as hf:
            keys = hf["key"][:]
            keys = keys.astype(feature_dtype) # in case mismatch of dtype between int and str
            pretrained_vocab = dict(zip(keys, range(len(keys))))
            pretrained_emb = hf["value"][:]
        # update vocab with pretrained keys, in case new token ids appear in validation or test set
        num_new_words = 0
        for word in pretrained_vocab.keys():
            if word not in self.vocab:
                self.vocab[word] = self.vocab.get("__PAD__", self.vocab_size) + num_new_words
                num_new_words += 1
        self.vocab_size += num_new_words
        if freeze_emb:
            embedding_matrix = np.zeros((self.vocab_size, embedding_dim))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(self.vocab_size, embedding_dim))
        if "__PAD__" in self.vocab:
            self.vocab["__PAD__"] = self.vocab_size - 1
            embedding_matrix[-1, :] = 0 # set as zero vector for PAD
        for word in pretrained_vocab.keys():
            embedding_matrix[self.vocab[word]] = pretrained_emb[pretrained_vocab[word]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset(feature_name, data=embedding_matrix)

    def load_vocab_from_file(self, vocab_file):
        with open(vocab_file, 'r') as fid:
            word_counts = json.load(fid)
        self.build_vocab(word_counts)

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab) + self.oov_token


def count_tokens(texts, splitter):
    word_counts = Counter()
    max_len = 0
    for text in texts:
        text_split = text.split(splitter)
        max_len = max(max_len, len(text_split))
        for token in text_split:
            word_counts[token] += 1
    return word_counts, max_len

        
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
            self.normalizer.fit(X.reshape(-1, 1))

    def normalize(self, X):
        if self.callable:
            return self.normalizer(X)
        else:
            return self.normalizer.transform(X.reshape(-1, 1)).flatten()


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
    """
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
