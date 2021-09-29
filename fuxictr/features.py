# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import io
import pickle
import os
import logging
import json
from collections import defaultdict
from .preprocess import Tokenizer


class FeatureMap(object):
    def __init__(self, dataset_id, version="pytorch"):
        self.dataset_id = dataset_id
        self.version = version
        self.num_fields = 0
        self.num_features = 0
        self.feature_len = 0
        self.feature_specs = OrderedDict()
        
    def set_feature_index(self):
        logging.info("Set feature index...")
        idx = 0
        for feature, feature_spec in self.feature_specs.items():
            if feature_spec["type"] != "sequence":
                self.feature_specs[feature]["index"] = idx
                idx += 1
            else:
                seq_indexes = [i + idx for i in range(feature_spec["max_len"])]
                self.feature_specs[feature]["index"] = seq_indexes
                idx += feature_spec["max_len"]
        self.feature_len = idx

    def get_feature_index(self, feature_type=None):
        feature_indexes = []
        if feature_type is not None:
            if not isinstance(feature_type, list):
                feature_type = [feature_type]
            feature_indexes = [feature_spec["index"] for feature, feature_spec in self.feature_specs.items()
                               if feature_spec["type"] in feature_type]
        return feature_indexes

    def load(self, json_file):
        logging.info("Load feature_map from json: " + json_file)
        with io.open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd, object_pairs_hook=OrderedDict)
        if feature_map["dataset_id"] != self.dataset_id:
            raise RuntimeError("dataset_id={} does not match to feature_map!".format(self.dataset_id))
        self.num_fields = feature_map["num_fields"]
        self.num_features = feature_map.get("num_features", None)
        self.feature_len = feature_map.get("feature_len", None)
        self.feature_specs = OrderedDict(feature_map["feature_specs"])

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["num_features"] = self.num_features
        feature_map["feature_len"] = self.feature_len
        feature_map["feature_specs"] = self.feature_specs
        with open(json_file, "w") as fd:
            json.dump(feature_map, fd, indent=4)


class FeatureEncoder(object):
    def __init__(self, 
                 feature_cols=[], 
                 label_col={}, 
                 dataset_id=None, 
                 data_root="../data/", 
                 version="pytorch", 
                 **kwargs):
        logging.info("Set up feature encoder...")
        self.data_dir = os.path.join(data_root, dataset_id)
        self.pickle_file = os.path.join(self.data_dir, "feature_encoder.pkl")
        self.json_file = os.path.join(self.data_dir, "feature_map.json")
        self.feature_cols = self._complete_feature_cols(feature_cols)
        self.label_col = label_col
        self.version = version
        self.feature_map = FeatureMap(dataset_id, version)
        self.encoders = dict()

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

    def read_csv(self, data_path):
        logging.info("Reading file: " + data_path)
        all_cols = self.feature_cols + [self.label_col]
        dtype_dict = dict((x["name"], eval(x["dtype"]) if isinstance(x["dtype"], str) else x["dtype"]) 
                          for x in all_cols)
        ddf = pd.read_csv(data_path, dtype=dtype_dict, memory_map=True) 
        return ddf

    def _preprocess(self, ddf):
        logging.info("Preprocess feature columns...")
        all_cols = [self.label_col] + self.feature_cols[::-1]
        for col in all_cols:
            name = col["name"]
            if name in ddf.columns and ddf[name].isnull().values.any():
                ddf[name] = self._fill_na(col, ddf[name])
            if "preprocess" in col and col["preprocess"] != "":
                preprocess_fn = getattr(self, col["preprocess"])
                ddf[name] = preprocess_fn(ddf, name)
        active_cols = [self.label_col["name"]] + [col["name"] for col in self.feature_cols if col["active"]]
        ddf = ddf.loc[:, active_cols]
        return ddf

    def _fill_na(self, col, series):
        na_value = col.get("na_value")
        if na_value is not None:
            return series.fillna(na_value)
        elif col["dtype"] == "str":
            return series.fillna("")
        else:
            raise RuntimeError("Feature column={} requires to assign na_value!".format(col["name"]))

    def fit(self, train_data, min_categr_count=1, num_buckets=10, **kwargs):           
        ddf = self.read_csv(train_data)
        ddf = self._preprocess(ddf)
        logging.info("Fit feature encoder...")
        self.feature_map.num_fields = 0
        for col in self.feature_cols:
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                name = col["name"]
                self.fit_feature_col(col, ddf, 
                                     min_categr_count=min_categr_count,
                                     num_buckets=num_buckets)
                self.feature_map.num_fields += 1
        self.feature_map.set_feature_index()
        self.save_pickle(self.pickle_file)
        self.feature_map.save(self.json_file)
        logging.info("Set feature encoder done.")
        
    def fit_feature_col(self, feature_column, ddf, min_categr_count=1, num_buckets=10):
        name = feature_column["name"]
        feature_type = feature_column["type"]
        feature_source = feature_column.get("source", "")
        self.feature_map.feature_specs[name] = {"source": feature_source,
                                                "type": feature_type}
        if "min_categr_count" in feature_column:
            min_categr_count = feature_column["min_categr_count"]
        self.feature_map.feature_specs[name]["min_categr_count"] = min_categr_count
        if "embedding_dim" in feature_column:
            self.feature_map.feature_specs[name]["embedding_dim"] = feature_column["embedding_dim"]
        feature_values = ddf[name].values
        if feature_type == "numeric":
            normalizer_name = feature_column.get("normalizer", None)
            if normalizer_name is not None:
                normalizer = Normalizer(normalizer_name)
                normalizer.fit(feature_values)
                self.encoders[name + "_normalizer"] = normalizer
            self.feature_map.num_features += 1
        elif feature_type == "categorical":
            encoder = feature_column.get("encoder", "")
            if encoder != "":
                self.feature_map.feature_specs[name]["encoder"] = encoder
            if encoder == "":
                tokenizer = Tokenizer(min_freq=min_categr_count, 
                                      na_value=feature_column.get("na_value", ""))
                if "share_embedding" in feature_column:
                    self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
                    tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
                else:
                    if self.is_share_embedding_with_sequence(name):
                        tokenizer.fit_on_texts(feature_values, use_padding=True)
                        self.feature_map.feature_specs[name]["padding_idx"] = tokenizer.vocab_size - 1
                    else:
                        tokenizer.fit_on_texts(feature_values, use_padding=False)
                self.encoders[name + "_tokenizer"] = tokenizer
                self.feature_map.num_features += tokenizer.vocab_size
                self.feature_map.feature_specs[name]["vocab_size"] = tokenizer.vocab_size
                if "pretrained_emb" in feature_column:
                    logging.info("Loading pretrained embedding: " + name)
                    self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_embedding.h5"
                    self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
                    tokenizer.load_pretrained_embedding(name,
                                                        feature_column["pretrained_emb"], 
                                                        feature_column["embedding_dim"],
                                                        os.path.join(self.data_dir, "pretrained_embedding.h5"))
            elif encoder == "numeric_bucket":
                num_buckets = feature_column.get("num_buckets", num_buckets)
                qtf = sklearn_preprocess.QuantileTransformer(n_quantiles=num_buckets + 1)
                qtf.fit(feature_values)
                boundaries = qtf.quantiles_[1:-1]
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_boundaries"] = boundaries
            elif encoder == "hash_bucket":
                num_buckets = feature_column.get("num_buckets", num_buckets)
                uniques = Counter(feature_values)
                num_buckets = min(num_buckets, len(uniques))
                self.feature_map.feature_specs[name]["vocab_size"] = num_buckets
                self.feature_map.num_features += num_buckets
                self.encoders[name + "_num_buckets"] = num_buckets
        elif feature_type == "sequence":
            encoder = feature_column.get("encoder", "MaskedAveragePooling")
            splitter = feature_column.get("splitter", " ")
            na_value = feature_column.get("na_value", "")
            max_len = feature_column.get("max_len", 0)
            padding = feature_column.get("padding", "post")
            tokenizer = Tokenizer(min_freq=min_categr_count, splitter=splitter, 
                                  na_value=na_value, max_len=max_len, padding=padding)
            if "share_embedding" in feature_column:
                self.feature_map.feature_specs[name]["share_embedding"] = feature_column["share_embedding"]
                tokenizer.set_vocab(self.encoders["{}_tokenizer".format(feature_column["share_embedding"])].vocab)
            else:
                tokenizer.fit_on_texts(feature_values, use_padding=True)
            self.encoders[name + "_tokenizer"] = tokenizer
            self.feature_map.num_features += tokenizer.vocab_size
            self.feature_map.feature_specs[name].update({"encoder": encoder,
                                                         "padding_idx": tokenizer.vocab_size - 1,
                                                         "vocab_size": tokenizer.vocab_size,
                                                         "max_len": tokenizer.max_len})
            if "pretrained_emb" in feature_column:
                logging.info("Loading pretrained embedding: " + name)
                self.feature_map.feature_specs[name]["pretrained_emb"] = "pretrained_embedding.h5"
                self.feature_map.feature_specs[name]["freeze_emb"] = feature_column.get("freeze_emb", True)
                tokenizer.load_pretrained_embedding(name,
                                                    feature_column["pretrained_emb"], 
                                                    feature_column["embedding_dim"],
                                                    os.path.join(self.data_dir, "pretrained_embedding.h5"))
        else:
            raise NotImplementedError("feature_col={}".format(feature_column))

    def transform(self, ddf):
        ddf = self._preprocess(ddf)
        logging.info("Transform feature columns...")
        data_arrays = []
        for feature, feature_spec in self.feature_map.feature_specs.items():
            feature_type = feature_spec["type"]
            if feature_type == "numeric":
                numeric_array = ddf.loc[:, feature].fillna(0).apply(lambda x: float(x)).values
                normalizer = self.encoders.get(feature + "_normalizer")
                if normalizer:
                     numeric_array = normalizer.normalize(numeric_array)
                data_arrays.append(numeric_array) 
            elif feature_type == "categorical":
                encoder = feature_spec.get("encoder", "")
                if encoder == "":
                    data_arrays.append(self.encoders.get(feature + "_tokenizer") \
                                                    .encode_category(ddf.loc[:, feature].values))
                elif encoder == "numeric_bucket":
                    raise NotImplementedError
                elif encoder == "hash_bucket":
                    raise NotImplementedError
            elif feature_type == "sequence":
                data_arrays.append(self.encoders.get(feature + "_tokenizer") \
                                                .encode_sequence(ddf.loc[:, feature].values))
        label_name = self.label_col["name"]
        if ddf[label_name].dtype != np.float64:
            ddf.loc[:, label_name] = ddf.loc[:, label_name].apply(lambda x: float(x))
        data_arrays.append(ddf.loc[:, label_name].values) # add the label column at last
        data_arrays = [item.reshape(-1, 1) if item.ndim == 1 else item for item in data_arrays]
        data_array = np.hstack(data_arrays)
        return data_array

    def is_share_embedding_with_sequence(self, feature):
        for col in self.feature_cols:
            if col.get("share_embedding", None) == feature and col["type"] == "sequence":
                return True
        return False

    def load_pickle(self, pickle_file=None):
        if pickle_file is None:
            pickle_file = self.pickle_file
        logging.info("Load feature_encoder from pickle: " + pickle_file)
        if os.path.exists(pickle_file):
            pickled_feature_encoder = pickle.load(open(pickle_file, "rb"))
            if pickled_feature_encoder.feature_map.dataset_id == self.feature_map.dataset_id:
                pickled_feature_encoder.version = self.version
                return pickled_feature_encoder
        raise IOError("pickle_file={} not valid.".format(pickle_file))

    def save_pickle(self, pickle_file):
        logging.info("Pickle feature_encode: " + pickle_file)
        if not os.path.exists(os.path.dirname(pickle_file)):
            os.makedirs(os.path.dirname(pickle_file))
        pickle.dump(self, open(pickle_file, "wb"))
        

    def load_json(self, json_file):
        self.feature_map.load(json_file)


