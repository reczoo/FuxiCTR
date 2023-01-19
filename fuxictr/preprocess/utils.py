# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import os
from tqdm import tqdm
import logging
import sklearn.preprocessing as sklearn_preprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


class Tokenizer(object):
    def __init__(self, num_IDs=None, na_value="", min_freq=1, splitter=None, remap=True,
                 lower=False, max_len=0, padding="pre", num_workers=8):
        self._num_IDs = num_IDs
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.vocab = dict()
        self.max_len = max_len
        self.padding = padding
        self.num_workers = num_workers
        self.remap = remap

    def fit_on_texts(self, texts):
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
        word_counts = word_counts.items()
        if self._num_IDs:
            word_counts = sorted(word_counts, key=lambda x: (-x[1], x[0]))
            word_counts = word_counts[0:self._num_IDs]
        words = []
        for token, count in word_counts:
            if count >= self._min_freq:
                if token != self._na_value:
                    words.append(token.lower() if self._lower else token)
        if self.remap:
            words.sort() # sort to guarantee the determinism of index order
            self.vocab = dict((token, idx) for idx, token in enumerate(words, 1))
        else:
            self.vocab = dict((token, int(token)) for token in words)
        self.vocab["__PAD__"] = 0 # use 0 for reserved __PAD__
        self.vocab["__OOV__"] = self.vocab_size() # use the last index for __OOV__

    def merge_vocab(self, shared_tokenizer):
        if self.remap:
            new_words = 0
            for word in self.vocab.keys():
                if word not in shared_tokenizer.vocab:
                    shared_tokenizer.vocab[word] = shared_tokenizer.vocab["__OOV__"] + new_words
                    new_words += 1
        else:
            shared_tokenizer.vocab.update(self.vocab)
        vocab_size = shared_tokenizer.vocab_size()
        if shared_tokenizer.vocab["__OOV__"] != vocab_size - 1:
            shared_tokenizer.vocab["__OOV__"] = vocab_size
        self.vocab = shared_tokenizer.vocab
        return shared_tokenizer

    def vocab_size(self):
        return max(self.vocab.values()) + 1

    def add_vocab(self, word_list):
        new_words = 0
        for word in word_list:
            if word not in self.vocab:
                self.vocab[word] = self.vocab["__OOV__"] + new_words
                new_words += 1
        if new_words > 0:
            self.vocab["__OOV__"] = self.vocab_size()

    def encode_meta(self, values):
        word_counts = Counter(list(values))
        if len(self.vocab) == 0:
            self.build_vocab(word_counts)
        else:
            self.add_vocab(word_counts.keys())
        meta_values = [self.vocab.get(x, self.vocab["__OOV__"]) for x in values]
        return np.array(meta_values)

    def encode_category(self, categories):
        category_indices = [self.vocab.get(x, self.vocab["__OOV__"]) for x in categories]
        return np.array(category_indices)

    def encode_sequence(self, texts):
        sequence_list = []
        for text in texts:
            if pd.isnull(text) or text == '':
                sequence_list.append([])
            else:
                sequence_list.append([self.vocab.get(x, self.vocab["__OOV__"]) if x != self._na_value \
                                      else self.vocab["__PAD__"] for x in text.split(self._splitter)])
        sequence_list = pad_sequences(sequence_list, maxlen=self.max_len, value=self.vocab["__PAD__"],
                                      padding=self.padding, truncating=self.padding)
        return np.array(sequence_list)
    
    def load_pretrained_embedding(self, feature_name, feature_dtype, pretrain_path,  
                                  output_path, freeze_emb=True):
        with h5py.File(pretrain_path, 'r') as hf:
            keys = hf["key"][:]
            keys = keys.astype(feature_dtype) # in case mismatch of dtype between int and str
            pretrained_vocab = dict(zip(keys, range(len(keys))))
            pretrained_emb = hf["value"][:]
        # update vocab with pretrained keys, in case new token ids appear in validation or test set
        self.add_vocab(pretrained_vocab.keys())
        
        logging.info("{}\'s pretrained_emb shape: {}".format(feature_name, pretrained_emb.shape))
        embedding_dim = pretrained_emb.shape[1]
        if freeze_emb:
            embedding_matrix = np.zeros((self.vocab_size(), embedding_dim))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(self.vocab_size(), embedding_dim))
            embedding_matrix[self.vocab["__PAD__"], :] = 0.  # set as zero embedding for PAD
        for word in pretrained_vocab.keys():
            embedding_matrix[self.vocab[word]] = pretrained_emb[pretrained_vocab[word]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'a') as hf: # Add different embeddings to a single h5 file
            hf.create_dataset(feature_name, data=embedding_matrix)

    # def load_vocab(self, vocab_file):
    #     with open(vocab_file, 'r') as fid:
    #         word_counts = json.load(fid)
    #     self.build_vocab(word_counts)


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
            null_index = np.isnan(X)
            self.normalizer.fit(X[~null_index].reshape(-1, 1))

    def normalize(self, X):
        if self.callable:
            return self.normalizer(X)
        else:
            return self.normalizer.transform(X.reshape(-1, 1)).flatten()


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length 
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