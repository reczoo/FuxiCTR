# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
import numpy as np
import h5py
from tqdm import tqdm
import polars as pl
from keras_preprocessing.sequence import pad_sequences
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class Tokenizer(object):
    def __init__(self, max_features=None, na_value="", min_freq=1, splitter=None, remap=True,
                 lower=False, max_len=0, padding="pre"):
        self._max_features = max_features
        self._na_value = na_value
        self._min_freq = min_freq
        self._lower = lower
        self._splitter = splitter
        self.vocab = dict()
        self.max_len = max_len
        self.padding = padding
        self.remap = remap

    def fit_on_texts(self, series):
        max_len = 0
        word_counts = Counter()
        with ProcessPoolExecutor(max_workers=(mp.cpu_count() // 2)) as executor:
            chunk_size = 1000000
            tasks = []
            for idx in range(0, len(series), chunk_size):
                data_chunk = series.iloc[idx: (idx + chunk_size)]
                tasks.append(executor.submit(count_tokens, data_chunk, self._splitter))
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                chunk_word_counts, chunk_max_len = future.result()
                word_counts.update(chunk_word_counts)
                max_len = max(max_len, chunk_max_len)
        if self.max_len == 0:  # if argument max_len not given
            self.max_len = max_len
        self.build_vocab(word_counts)

    def build_vocab(self, word_counts):
        word_counts = word_counts.items()
        # sort to guarantee the determinism of index order
        word_counts = sorted(word_counts, key=lambda x: (-x[1], x[0]))
        if self._max_features: # keep the most frequent features
            word_counts = word_counts[0:self._max_features]
        words = []
        for token, count in word_counts:
            if count >= self._min_freq:
                if token != self._na_value:
                    words.append(token.lower() if self._lower else token)
            else:
                break # already sorted in decending order
        if self.remap:
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
        if (shared_tokenizer.vocab["__OOV__"] != vocab_size - 1 or
            shared_tokenizer.vocab["__OOV__"] != len(shared_tokenizer.vocab) - 1):
            shared_tokenizer.vocab["__OOV__"] = vocab_size
        self.vocab = shared_tokenizer.vocab
        return shared_tokenizer

    def vocab_size(self):
        return max(self.vocab.values()) + 1 # In case that keys start from 1

    def update_vocab(self, word_list):
        new_words = 0
        for word in word_list:
            if word not in self.vocab:
                self.vocab[word] = self.vocab.get("__OOV__", 0) + new_words
                new_words += 1
        if new_words > 0:
            self.vocab["__OOV__"] = self.vocab_size()

    def encode_meta(self, series):
        word_counts = dict(series.value_counts())
        if len(self.vocab) == 0:
            self.build_vocab(word_counts)
        else: # for considering meta data in test data
            self.update_vocab(word_counts.keys())
        series = series.map(lambda x: self.vocab.get(x, self.vocab["__OOV__"]))
        return series.values

    def encode_category(self, series):
        series = series.map(lambda x: self.vocab.get(x, self.vocab["__OOV__"]))
        return series.values

    def encode_sequence(self, series):
        series = series.map(
            lambda text: [self.vocab.get(x, self.vocab["__OOV__"]) if x != self._na_value \
            else self.vocab["__PAD__"] for x in text.split(self._splitter)]
        )
        seqs = pad_sequences(series.to_list(), maxlen=self.max_len,
                             value=self.vocab["__PAD__"],
                             padding=self.padding, truncating=self.padding)
        return seqs.tolist()
    
    def load_pretrained_vocab(self, feature_dtype, pretrain_path, expand_vocab=True):
        keys = load_pretrain_emb(pretrain_path, keys=["key"])
        # in case mismatch of dtype between int and str
        keys = keys.astype(feature_dtype)
        # Update vocab with pretrained keys in case new tokens appear in validation or test set
        # Do NOT update OOV index here since it is used in PretrainedEmbedding
        if expand_vocab:
            vocab_size = self.vocab_size()
            for word in keys:
                if word not in self.vocab:
                    self.vocab[word] = vocab_size
                    vocab_size += 1


def count_tokens(series, splitter=None):
    max_len = 0
    if splitter is not None: # for sequence
        series = series.map(lambda text: text.split(splitter))
        max_len = series.str.len().max()
        word_counts = series.explode().value_counts()
    else:
        word_counts = series.value_counts()
    return dict(word_counts), max_len


def load_pretrain_emb(pretrain_path, keys=["key", "value"]):
    if type(keys) != list:
        keys = [keys]
    if pretrain_path.endswith("h5"):
        with h5py.File(pretrain_path, 'r') as hf:
            values = [hf[k][:] for k in keys]
    elif pretrain_path.endswith("npz"):
        npz = np.load(pretrain_path)
        values = [npz[k] for k in keys]
    elif pretrain_path.endswith("parquet"):
        df = pd.read_parquet(pretrain_path)
        values = [df[k].values for k in keys]
    else:
        raise ValueError(f"Embedding format not supported: {pretrain_path}")
    return values[0] if len(values) == 1 else values
