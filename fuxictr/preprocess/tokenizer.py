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
import polars as pl


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences (list of list) to the ndarray of same length.
    This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences.
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

    def build_vocab(self, word_counts):
        """Build vocabulary from token frequency counts.

        Args:
            word_counts (Counter or dict): Token frequency counts. Both
                ``collections.Counter`` and a plain ``dict`` are supported.
        """
        word_counts = Counter(word_counts)
        word_counts = word_counts.most_common() # sort to guarantee the determinism of index order
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
        # sort to guarantee the determinism of index order for new words
        for word in sorted(word_list):
            if word not in self.vocab:
                self.vocab[word] = self.vocab.get("__OOV__", 0) + new_words
                new_words += 1
        if new_words > 0:
            self.vocab["__OOV__"] = self.vocab_size() # __OOV__ move to max + 1
    
    def encode_category(self, series):
        """Encode a categorical column to integer indices.

        Args:
            series (list or pandas.Series): Raw categorical values.

        Returns:
            list: Encoded integer values.
        """
        return list(map(lambda x: self.vocab.get(x, self.vocab["__OOV__"]), series))

    def encode_sequence(self, series):
        """Encode a sequence column to padded integer arrays.

        Args:
            series (list or pandas.Series): Raw sequence strings.

        Returns:
            numpy.ndarray: 2D padded integer array of shape (n_rows, maxlen).
        """
        seqs = list(map(
            lambda text: [self.vocab.get(x, self.vocab["__OOV__"]) if x != self._na_value
                          else self.vocab["__PAD__"] for x in text.split(self._splitter)],
            series
        ))
        seqs = pad_sequences(seqs, maxlen=self.max_len,
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


def load_pretrain_emb(pretrain_path, keys=["key", "value"]):
    """Load pretrained embedding data from file.

    Supports ``.h5``, ``.npz``, and ``.parquet`` formats.

    Args:
        pretrain_path (str): Path to embedding file.
        keys (list): Keys to read from the file. Default: ``["key", "value"]``.

    Returns:
        numpy.ndarray or tuple: Loaded embedding data.

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
        values = [df[k].to_numpy() for k in keys]
    else:
        raise ValueError(f"Embedding format not supported: {pretrain_path}")
    return values[0] if len(values) == 1 else values
