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
from typing import Iterable
import numpy as np
import h5py
from tqdm import tqdm
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class Tokenizer(object):
    """Tokenizes text data and builds vocabularies for categorical or sequence features.

    Supports building vocab from raw text, merging with shared tokenizers,
    and encoding text into integer indices with optional padding for sequences.

    Args:
        max_features (int, optional): Maximum vocabulary size. Default: ``None``.
        na_value (str): String value treated as missing/NA. Default: ``\"\"``.
        min_freq (int): Minimum token frequency to include in vocabulary. Default: ``1``.
        splitter (str, optional): Delimiter for sequence splitting. Default: ``None``.
        remap (bool): If ``True``, remap tokens to consecutive integer indices.
            Default: ``True``.
        lower (bool): If ``True``, lowercase tokens. Default: ``False``.
        max_len (int): Maximum sequence length. ``0`` means auto-detect. Default: ``0``.
        padding (str): ``"pre"`` or ``"post"`` padding for sequences. Default: ``"pre"``.
    """

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
        """Fit tokenizer on a text series and build vocabulary.

        Uses parallel processing to count tokens across chunks.

        Args:
            series (pandas.Series): Text data series.
        """
        max_len = 0
        word_counts = Counter()
        with ProcessPoolExecutor(max_workers=(mp.cpu_count() // 2)) as executor:
            chunk_size = 1000000
            tasks = []
            for idx in range(0, len(series), chunk_size):
                data_chunk = series.slice(idx, chunk_size)
                tasks.append(executor.submit(count_tokens, data_chunk, self._splitter))
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                chunk_word_counts, chunk_max_len = future.result()
                word_counts.update(chunk_word_counts)
                max_len = max(max_len, chunk_max_len)
        if self.max_len == 0:  # if argument max_len not given
            self.max_len = max_len
        self.build_vocab(word_counts)

    def build_vocab(self, word_counts):
        """Build vocabulary from token frequency counts.

        Args:
            word_counts (Counter): Token frequency counts.
        """
        # sort to guarantee the determinism of index order
        word_counts = word_counts.most_common()
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
        """Merge this tokenizer's vocabulary into a shared tokenizer.

        Args:
            shared_tokenizer (Tokenizer): Tokenizer to merge into.

        Returns:
            Tokenizer: The updated shared tokenizer.
        """
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
        """Return the vocabulary size.

        Returns:
            int: Size of the vocabulary (max index + 1).
        """
        return max(self.vocab.values()) + 1 # In case that keys start from 1

    def update_vocab(self, word_list):
        """Update vocabulary with new words.

        Args:
            word_list (iterable): Words to add.
        """
        new_words = 0
        for word in word_list:
            if word not in self.vocab:
                self.vocab[word] = self.vocab.get("__OOV__", 0) + new_words
                new_words += 1
        if new_words > 0:
            self.vocab["__OOV__"] = self.vocab_size()

    def encode_meta(self, series):
        """Encode a meta column series to integer indices.

        Args:
            series (polars.Series): Raw meta values.

        Returns:
            polars.Series: Encoded integer values.
        """
        word_counts = dict(series.value_counts())
        if len(self.vocab) == 0:
            self.build_vocab(word_counts)
        else: # for considering meta data in test data
            self.update_vocab(word_counts.keys())

        series = self.encode_category(series)
        return series

    def encode_category(self, series):
        """Encode a categorical series to integer indices.

        Args:
            series (polars.Series): Raw categorical values.

        Returns:
            polars.Series: Encoded integer values.
        """
        vocab = {key: self.vocab[key] for key in set(series.unique()) & set(self.vocab.keys())}
        # polars complains if vocab keys are of different type than series (ie "__PAD__" and numeric series)
        series = series.replace_strict(vocab, default=self.vocab["__OOV__"])
        return series

    def encode_sequence(self, series):
        """Encode a sequence series to padded integer arrays.

        Args:
            series (polars.Series): Raw sequence strings.

        Returns:
            series (polars.Series): padded integer sequences.
        """
        
        series = (
            series.str.split(self._splitter)
            .list.eval(
                pl.when(pl.element()!=self._na_value)
                .then(pl.element().replace_strict(self.vocab,default=self.vocab["__OOV__"]))
                .otherwise(self.vocab["__PAD__"]))
        )
        seqs = pad_sequences(series.to_list(), maxlen=self.max_len,
                              value=self.vocab["__PAD__"],
                              padding=self.padding, truncating=self.padding)
        return seqs

    def load_pretrained_vocab(self, feature_dtype, pretrain_path, expand_vocab=True):
        """Load pretrained embedding keys and optionally expand vocabulary.

        Args:
            feature_dtype (type): Data type for feature keys.
            pretrain_path (str): Path to pretrained embedding file.
            expand_vocab (bool): Whether to add new keys to vocabulary. Default: ``True``.
        """
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


def pad_sequences(sequences: Iterable[Iterable[int]], maxlen=None,
                  padding='pre', truncating='pre', value=0.):
    if not isinstance(sequences,pl.Series):
        sequences = pl.Series(sequences)
    sequence_lengths = sequences.list.len()
    if maxlen is None:
        maxlen = sequence_lengths.max()
    sequence_lengths = sequence_lengths.clip(upper_bound=maxlen)
    if truncating == 'pre':
        sequences = sequences.list.slice(0, maxlen)
    elif truncating == 'post': 
        sequences = sequences.list.slice(-maxlen)
    else:
        raise ValueError(f'Truncating type "{truncating}" not understood')
    padder = pl.select(pl.repeat(value,len(sequences)).repeat_by(maxlen - sequence_lengths).alias("sequence")).to_series()
    # convert to type
    # test for 0 repeat
    # sample_shape?
    if padding == 'pre':
        sequences = padder.list.concat(sequences)
    elif padding == 'post':
        sequences = sequences.list.concat(padder)
    else: 
        raise ValueError(f'Padding type "{padding}" not understood')
    sequences = sequences.list.to_array(maxlen) # can be converted to 2d numpy array by .to_numpy()
    return sequences
    

def count_tokens(series, splitter=None):
    """Count token frequencies and max sequence length in a series.

    Args:
        series (polars.Series): Text data series.
        splitter (str, optional): Delimiter for splitting sequences.

    Returns:
        tuple: ``(word_counts, max_len)`` where ``word_counts`` is a dict
        and ``max_len`` is the maximum sequence length.
    """
    max_len = 0
    if splitter is not None: # for sequence
        series = series.str.split(splitter)
        max_len = series.list.len().max()
        word_counts = series.list.explode().value_counts()
    else:
        word_counts = series.value_counts()
    return dict(word_counts.iter_rows()), max_len


def load_pretrain_emb(pretrain_path, keys=["key", "value"]):
    """Load pretrained embedding data from file.

    Supports ``.h5``, ``.npz``, and ``.parquet`` formats.

    Args:
        pretrain_path (str): Path to embedding file.
        keys (list): Keys to read from the file. Default: ``["key", "value"]``.

    Returns:
        numpy.ndarray if single embedding else list[numpy.ndarray]: Loaded embedding data.

    Raises:
        ValueError: If the file format is not supported.
    """
    if type(keys) != list:
        keys = [keys]
    if pretrain_path.endswith("h5"):
        with h5py.File(pretrain_path, 'r') as hf:
            values = [hf[k][:] for k in keys]
    elif pretrain_path.endswith("npz"):
        npz = np.load(pretrain_path)
        values = [npz[k] for k in keys]
    elif pretrain_path.endswith("parquet"):
        df = pl.read_parquet(pretrain_path)
        values = [df.get_column(k).to_numpy() for k in keys]
    else:
        raise ValueError(f"Embedding format not supported: {pretrain_path}")
    return values[0] if len(values) == 1 else values
