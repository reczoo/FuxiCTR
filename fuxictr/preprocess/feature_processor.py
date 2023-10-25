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


import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import pickle
import os
import logging
import json
import re
import shutil
import sklearn.preprocessing as sklearn_preprocess
from fuxictr.features import FeatureMap
from .tokenizer import Tokenizer
from .normalizer import Normalizer


class FeatureProcessor(object):
    def __init__(self,
                 feature_cols=[],
                 label_col=[],
                 dataset_id=None, 
                 data_root="../data/",
                 **kwargs):
        logging.info("Set up feature processor...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_processor.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.vocab_file = os.path.join(self.data_dir, "feature_vocab.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_cols = label_col if type(label_col) == list else [label_col]
        self.feature_map = FeatureMap(dataset_id, self.data_dir)
        self.feature_map.labels = [col["name"] for col in self.label_cols]
        self.feature_map.group_id = kwargs.get("group_id", None)
        self.dtype_dict = dict((feat["name"], eval(feat["dtype"]) if type(feat["dtype"]) == str else feat["dtype"]) 
                                for feat in self.feature_cols + self.label_cols)
        self.processor_dict = dict()

    def _complete_feature_cols(self, feature_cols):
        full_feature_cols = []
        for col in feature_cols:
            name_or_namelist = col["name"]
            if isinstance(name_or_namelist, list):
                for _name in name_or_namelist:
                    _col = col.copy()
                    _col["name"] = _name
                    full_feature_cols.append(_col)
            else:
                full_feature_cols.append(col)
        return full_feature_cols

    def read_csv(self, data_path, sep=",", nrows=None, **kwargs):
        logging.info("Reading file: " + data_path)
        usecols_fn = lambda x: x in self.dtype_dict
        ddf = pd.read_csv(data_path, sep=sep, usecols=usecols_fn, 
                          dtype=object, memory_map=True, nrows=nrows)
        return ddf

    def preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        all_cols = self.label_cols + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            if name in ddf.columns:
                if ddf[name].isnull().values.any():
                    ddf[name] = self._fill_na_(col, ddf[name])
                ddf[name] = ddf[name].astype(self.dtype_dict[name])
            if col.get("preprocess"):
                preprocess_splits = re.split(r"\(|\)", col["preprocess"])
                preprocess_fn = getattr(self, preprocess_splits[0])
                if len(preprocess_splits) > 1:
                    ddf[name] = preprocess_fn(ddf, preprocess_splits[1])
                else:
                    ddf[name] = preprocess_fn(ddf, name)
                ddf[name] = ddf[name].astype(self.dtype_dict[name])
        active_cols = [col["name"] for col in all_cols if col.get("active") != False]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na_(self, col, series):
        na_value = col.get("fill_na")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] in ["str", str]:
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign fill_na value!".format(col["name"]))

    def fit(self, train_ddf, min_categr_count=1, num_buckets=10, **kwargs):    
        logging.info("Fit feature processor...")
        for col in self.feature_cols:
            name = col["name"]
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                if col["type"] == "meta": # e.g. group_id
                    self.fit_meta_col(col, train_ddf[name].values)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, train_ddf[name].values)
                elif col["type"] == "categorical":
                    self.fit_categorical_col(col, train_ddf[name].values, 
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, train_ddf[name].values, 
                                          min_categr_count=min_categr_count)
                else:
                    raise NotImplementedError("feature_col={}".format(feature_col))
        
        # Expand vocab from pretrained_emb
        os.makedirs(self.data_dir, exist_ok=True)
        for col in self.feature_cols:
            name = col["name"]
            if "pretrained_emb" in col:
                logging.info("Loading pretrained embedding: " + name)
                if "pretrain_dim" in col:
                    self.feature_map.features[name]["pretrain_dim"] = col["pretrain_dim"]
                shutil.copy(col["pretrained_emb"],
                            os.path.join(self.data_dir, "pretrained_{}".format(name)))
                self.feature_map.features[name]["pretrained_emb"] = "pretrained_{}.h5".format(name)
                self.feature_map.features[name]["freeze_emb"] = col.get("freeze_emb", True)
                self.feature_map.features[name]["pretrain_usage"] = col.get("pretrain_usage", "init")
                tokenizer = self.processor_dict[name + "::tokenizer"]
                tokenizer.load_pretrained_vocab(self.dtype_dict[name], col["pretrained_emb"])
                self.processor_dict[name + "::tokenizer"] = tokenizer
                self.feature_map.features[name]["vocab_size"] = tokenizer.vocab_size()

        # Handle share_embedding vocab re-assign
        for name, spec in self.feature_map.features.items():
            if spec["type"] == "numeric":
                self.feature_map.total_features += 1
            elif spec["type"] in ["categorical", "sequence"]:
                if "share_embedding" in spec:
                    # sync vocab from the shared_embedding field
                    tokenizer = self.processor_dict[name + "::tokenizer"]
                    tokenizer.vocab = self.processor_dict[spec["share_embedding"] + "::tokenizer"].vocab
                    self.processor_dict[name + "::tokenizer"] = tokenizer
                    self.feature_map.features[name].update({"oov_idx": tokenizer.vocab["__OOV__"],
                                                            "vocab_size": tokenizer.vocab_size()})
                else:
                    self.feature_map.total_features += self.feature_map.features[name]["vocab_size"]
        self.feature_map.num_fields = self.feature_map.get_num_fields()
        self.feature_map.set_column_index()
        self.save_pickle(self.pickle_file)
        self.save_vocab(self.vocab_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature processor done.")

    def fit_meta_col(self, col, col_values):
        name = col["name"]
        feature_type = col["type"]
        self.feature_map.features[name] = {"type": feature_type}
        # assert col.get("remap") == False, "Meta feature currently only supports `remap=False`, \
            # since it needs to map train and validation sets together."
        if col.get("remap", True):
            tokenizer = Tokenizer(min_freq=1, remap=True)
            self.processor_dict[name + "::tokenizer"] = tokenizer

    def fit_numeric_col(self, col, col_values):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                                "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "normalizer" in col:
            normalizer = Normalizer(col["normalizer"])
            normalizer.fit(col_values)
            self.processor_dict[name + "::normalizer"] = normalizer

    def fit_categorical_col(self, col, col_values, min_categr_count=1, num_buckets=10):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        min_categr_count = col.get("min_categr_count", min_categr_count)
        self.feature_map.features[name] = {"source": feature_source,
                                                "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "emb_output_dim" in col:
            self.feature_map.features[name]["emb_output_dim"] = col["emb_output_dim"]
        if "category_processor" not in col:
            tokenizer = Tokenizer(min_freq=min_categr_count, 
                                  na_value=col.get("fill_na", ""), 
                                  remap=col.get("remap", True))
            tokenizer.fit_on_texts(col_values)
            if "share_embedding" in col:
                self.feature_map.features[name]["share_embedding"] = col["share_embedding"]
                tknzr_name = col["share_embedding"] + "::tokenizer"
                # update vocab of both tokenizers
                self.processor_dict[tknzr_name] = tokenizer.merge_vocab(self.processor_dict[tknzr_name])
                self.feature_map.features[col["share_embedding"]] \
                                .update({"oov_idx": self.processor_dict[tknzr_name].vocab["__OOV__"],
                                         "vocab_size": self.processor_dict[tknzr_name].vocab_size()})
            self.processor_dict[name + "::tokenizer"] = tokenizer
            self.feature_map.features[name].update({"padding_idx": 0,
                                                    "oov_idx": tokenizer.vocab["__OOV__"],
                                                    "vocab_size": tokenizer.vocab_size()})
        else:
            category_processor = col["category_processor"]
            self.feature_map.features[name]["category_processor"] = category_processor
            if category_processor == "quantile_bucket": # transform numeric value to bucket
                num_buckets = col.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                qtf.fit(col_values)
                boundaries = qtf.quantiles_[1:-1]
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::boundaries"] = boundaries
            elif category_processor == "hash_bucket":
                num_buckets = col.get("num_buckets", num_buckets)
                uniques = Counter(col_values)
                num_buckets = min(num_buckets, len(uniques))
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::num_buckets"] = num_buckets
            else:
                raise NotImplementedError("category_processor={} not supported.".format(category_processor))

    def fit_sequence_col(self, col, col_values, min_categr_count=1):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        min_categr_count = col.get("min_categr_count", min_categr_count)
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        feature_encoder = col.get("feature_encoder", "layers.MaskedAveragePooling()")
        if feature_encoder not in [None, "null", "None", "none"]:
            self.feature_map.features[name]["feature_encoder"] = feature_encoder
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "emb_output_dim" in col:
            self.feature_map.features[name]["emb_output_dim"] = col["emb_output_dim"]
        splitter = col.get("splitter")
        na_value = col.get("fill_na", "")
        max_len = col.get("max_len", 0)
        padding = col.get("padding", "post") # "post" or "pre"
        tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                              na_value=na_value, max_len=max_len, padding=padding,
                              remap=col.get("remap", True))
        tokenizer.fit_on_texts(col_values)
        if "share_embedding" in col:
            self.feature_map.features[name]["share_embedding"] = col["share_embedding"]
            tknzr_name = col["share_embedding"] + "::tokenizer"
            # update vocab of both tokenizers
            self.processor_dict[tknzr_name] = tokenizer.merge_vocab(self.processor_dict[tknzr_name])
            self.feature_map.features[col["share_embedding"]] \
                            .update({"oov_idx": self.processor_dict[tknzr_name].vocab["__OOV__"],
                                     "vocab_size": self.processor_dict[tknzr_name].vocab_size()})
        self.processor_dict[name + "::tokenizer"] = tokenizer
        self.feature_map.features[name].update({"padding_idx": 0,
                                                "oov_idx": tokenizer.vocab["__OOV__"],
                                                "max_len": tokenizer.max_len,
                                                "vocab_size": tokenizer.vocab_size()})

    def transform(self, ddf):
        logging.info("Transform feature columns...")
        data_dict = dict()
        for feature, feature_spec in self.feature_map.features.items():
            if feature in ddf.columns:
                feature_type = feature_spec["type"]
                col_values = ddf.loc[:, feature].values
                if feature_type == "meta":
                    if feature + "::tokenizer" in self.processor_dict:
                        tokenizer = self.processor_dict[feature + "::tokenizer"]
                        data_dict[feature] = tokenizer.encode_meta(col_values)
                        # Update vocab in tokenizer
                        self.processor_dict[feature + "::tokenizer"] = tokenizer
                    else:
                        data_dict[feature] = col_values.astype(self.dtype_dict[feature])
                elif feature_type == "numeric":
                    col_values = col_values.astype(float)
                    normalizer = self.processor_dict.get(feature + "::normalizer")
                    if normalizer:
                         col_values = normalizer.transform(col_values)
                    data_dict[feature] = col_values
                elif feature_type == "categorical":
                    category_processor = feature_spec.get("category_processor")
                    if category_processor is None:
                        data_dict[feature] = self.processor_dict.get(feature + "::tokenizer").encode_category(col_values)
                    elif category_processor == "numeric_bucket":
                        raise NotImplementedError
                    elif category_processor == "hash_bucket":
                        raise NotImplementedError
                elif feature_type == "sequence":
                    data_dict[feature] = self.processor_dict.get(feature + "::tokenizer").encode_sequence(col_values)
        for label in self.feature_map.labels:
            if label in ddf.columns:
                data_dict[label] = ddf.loc[:, label].values
        return data_dict

    def load_pickle(self, pickle_file=None):
        """ Load feature processor from cache """
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_processor from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_processor = pickle.load(open(pickle_file, "rb"))
            if pickled_feature_processor.feature_map.dataset_id == self.feature_map.dataset_id:
                return pickled_feature_processor
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encode: " + pickle_file)
        pickle.dump(self, open(pickle_file, "wb"))

    def save_vocab(self, vocab_file):
        logging.info("Save feature_vocab to json: " + vocab_file)
        vocab = dict()
        for feature, spec in self.feature_map.features.items():
            if spec["type"] in ["categorical", "sequence"]:
                vocab[feature] = OrderedDict(
                    sorted(self.processor_dict[feature + "::tokenizer"].vocab.items(), key=lambda x:x[1]))
        with open(vocab_file, "w") as fd:
            fd.write(json.dumps(vocab, indent=4))

    def copy_from(self, ddf, src_name):
        return ddf[src_name]

