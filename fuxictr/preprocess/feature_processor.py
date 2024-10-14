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


import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import polars as pl
import pickle
import os
import logging
import json
import re
import shutil
import glob
from pathlib import Path
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
        self.dtype_dict = dict(
            (feat["name"], eval(feat["dtype"]) if type(feat["dtype"]) == str else feat["dtype"]) 
            for feat in self.feature_cols + self.label_cols
        )
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

    def read_data(self, data_path, data_format="csv", sep=",", n_rows=None, **kwargs):
        if not data_path.endswith(data_format):
            data_path = os.path.join(data_path, f"*.{data_format}")
        logging.info("Reading files: " + data_path)
        file_names = sorted(glob.glob(data_path))
        assert len(file_names) > 0, f"Invalid data path: {data_path}"
        if data_format == "csv":
            dfs = [
                pl.scan_csv(source=file_name, separator=sep, dtypes=self.dtype_dict,
                            low_memory=False, n_rows=n_rows)
                for file_name in file_names
            ]
            ddf = pl.concat(dfs)
        elif data_format == "parquet":
            dfs = [
                pl.scan_parquet(source=file_name, low_memory=False, n_rows=n_rows)
                for file_name in file_names
            ]
            ddf = pl.concat(dfs)
            seq_cols = [x for x in ddf.columns if isinstance(ddf.select(x).dtypes[0], pl.List)]
            for col in seq_cols:
                # Convert list to "^" seperated string for the same preprocessing as csv format
                ddf = ddf.with_columns(pl.col(col).apply(lambda x: "^".join(map(str, x))))
        else:
            NotImplementedError(f"data_format={data_format} not supported.")
        return ddf

    def preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        all_cols = self.label_cols + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            fill_na = col.get("fill_na", 
                              "" if col["dtype"] in ["str", str] else 0)
            col_exist = name in ddf.columns
            if col_exist:
                ddf = ddf.with_columns(pl.col(name).fill_null(fill_na))
            if col.get("preprocess"):
                preprocess_args = re.split(r"\(|\)", col["preprocess"])
                preprocess_fn = getattr(self, preprocess_args[0])
                if len(preprocess_args) == 1:
                    preprocess_args = [name] # use col_name as args when not being explicitly set
                else:
                    preprocess_args = preprocess_args[1:-1]
                ddf = ddf.with_columns(
                    preprocess_fn(*preprocess_args)
                    .alias(name)
                    .cast(self.dtype_dict[name])
                )
            if not col_exist:
                ddf = ddf.with_columns(pl.col(name).fill_null(fill_na))
        active_cols = [col["name"] for col in all_cols if col.get("active") != False]
        ddf = ddf.select(active_cols)
        return ddf

    def fit(self, train_ddf, min_categr_count=1, num_buckets=10, rebuild_dataset=True, **kwargs):
        logging.info("Fit feature processor...")
        self.rebuild_dataset = rebuild_dataset
        for col in self.feature_cols:
            name = col["name"]
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                col_series = (
                    train_ddf.select(name).collect().to_series().to_pandas() if self.rebuild_dataset
                    else None
                )
                if col["type"] == "meta": # e.g. set group_id in gAUC
                    self.fit_meta_col(col)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, col_series)
                elif col["type"] == "embedding":
                    self.fit_embedding_col(col)
                elif col["type"] == "categorical":
                    self.fit_categorical_col(col, col_series,
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, col_series,
                                          min_categr_count=min_categr_count)
                else:
                    raise NotImplementedError("feature type={}".format(col["type"]))
        
        # Expand vocab from pretrained_emb
        os.makedirs(self.data_dir, exist_ok=True)
        for col in self.feature_cols:
            name = col["name"]
            if "pretrained_emb" in col:
                logging.info("Loading pretrained embedding: " + name)
                if "pretrain_dim" in col:
                    self.feature_map.features[name]["pretrain_dim"] = col["pretrain_dim"]
                ext = Path(col["pretrained_emb"]).suffix
                shutil.copy(col["pretrained_emb"],
                            os.path.join(self.data_dir, "pretrained_{}{}".format(name, ext)))
                self.feature_map.features[name]["pretrained_emb"] = "pretrained_{}{}".format(name, ext)
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
                if "pretrained_emb" not in spec: # "oov_idx" not used without pretrained_emb
                    del self.feature_map.features[name]["oov_idx"]

        self.feature_map.num_fields = self.feature_map.get_num_fields()
        self.feature_map.set_column_index()
        self.feature_map.save(self.json_file)
        self.save_pickle(self.pickle_file)
        self.save_vocab(self.vocab_file)
        logging.info("Set feature processor done.")

    def fit_meta_col(self, col):
        name = col["name"]
        feature_type = col["type"]
        self.feature_map.features[name] = {"type": feature_type}
        if col.get("remap", True):
            # No need to fit, update vocab in encode_meta()
            tokenizer = Tokenizer(min_freq=1, remap=True)
            self.processor_dict[name + "::tokenizer"] = tokenizer

    def fit_numeric_col(self, col, col_series):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "normalizer" in col:
            normalizer = Normalizer(col["normalizer"])
            if self.rebuild_dataset:
                normalizer.fit(col_series.dropna().values)
            self.processor_dict[name + "::normalizer"] = normalizer

    def fit_embedding_col(self, col):
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]

    def fit_categorical_col(self, col, col_series, min_categr_count=1, num_buckets=10):
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
            if self.rebuild_dataset:
                tokenizer.fit_on_texts(col_series)
            else:
                if "vocab_size" in col:
                    tokenizer.update_vocab(range(col["vocab_size"] - 1))
                else:
                    raise ValueError(f"{name}: vocab_size is required when rebuild_dataset=False")
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
                if self.rebuild_dataset:
                    qtf.fit(col_series.values)
                    boundaries = qtf.quantiles_[1:-1]
                    self.processor_dict[name + "::boundaries"] = boundaries
                self.feature_map.features[name]["vocab_size"] = num_buckets
            elif category_processor == "hash_bucket":
                num_buckets = col.get("num_buckets", num_buckets)
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::num_buckets"] = num_buckets
            else:
                raise NotImplementedError("category_processor={} not supported.".format(category_processor))

    def fit_sequence_col(self, col, col_series, min_categr_count=1):
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
        splitter = col.get("splitter", "^")
        na_value = col.get("fill_na", "")
        max_len = col.get("max_len", 0)
        padding = col.get("padding", "post") # "post" or "pre"
        tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                              na_value=na_value, max_len=max_len, padding=padding,
                              remap=col.get("remap", True))
        if self.rebuild_dataset:
            tokenizer.fit_on_texts(col_series)
        else:
            if "vocab_size" in col:
                tokenizer.update_vocab(range(col["vocab_size"] - 1))
            else:
                raise ValueError(f"{name}: vocab_size is required when rebuild_dataset=False")
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
        logging.info("Transform feature columns to IDs...")
        for feature, feature_spec in self.feature_map.features.items():
            if feature in ddf.columns:
                feature_type = feature_spec["type"]
                col_series = ddf[feature]
                if feature_type == "meta":
                    if feature + "::tokenizer" in self.processor_dict:
                        tokenizer = self.processor_dict[feature + "::tokenizer"]
                        ddf[feature] = tokenizer.encode_meta(col_series)
                        # Update vocab in tokenizer
                        self.processor_dict[feature + "::tokenizer"] = tokenizer
                elif feature_type == "numeric":
                    normalizer = self.processor_dict.get(feature + "::normalizer")
                    if normalizer:
                        ddf[feature] = normalizer.transform(col_series.values)
                elif feature_type == "categorical":
                    category_processor = feature_spec.get("category_processor")
                    if category_processor is None:
                        ddf[feature] = (
                            self.processor_dict.get(feature + "::tokenizer")
                            .encode_category(col_series)
                        )
                    elif category_processor == "numeric_bucket":
                        raise NotImplementedError
                    elif category_processor == "hash_bucket":
                        raise NotImplementedError
                elif feature_type == "sequence":
                    ddf[feature] = (self.processor_dict.get(feature + "::tokenizer")
                                    .encode_sequence(col_series))
        return ddf

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

    def copy_from(self, src_col):
        return pl.col(src_col)
