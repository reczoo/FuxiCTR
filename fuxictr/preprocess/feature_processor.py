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
    """Orchestrates data reading, preprocessing, fitting, and transformation for CTR datasets.

    ``FeatureProcessor`` manages feature columns (numeric, categorical, sequence,
    embedding, meta) and builds a ``FeatureMap`` together with fitted preprocessors
    such as tokenizers and normalizers.

    Args:
        feature_cols (list): List of feature column specification dicts. Default: ``[]``.
        label_col (list or dict): Label column specification(s). Default: ``[]``.
        dataset_id (str, optional): Unique dataset identifier. Default: ``None``.
        data_root (str): Root directory under which dataset folders are created. Default: ``"../data/"``.
        **kwargs: Additional keyword arguments (e.g. ``group_id``).
    """

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
        """Read raw data files lazily using Polars.

        Args:
            data_path (str): Path to a file or directory pattern.
            data_format (str): File format, either ``"csv"`` or ``"parquet"``.
            sep (str): Separator for CSV files.
            n_rows (int, optional): Maximum rows to read per file.
            **kwargs: Extra arguments forwarded to the reader.

        Returns:
            pl.LazyFrame: Concatenated lazy frame over all matched files.
        """
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
        else:
            NotImplementedError(f"data_format={data_format} not supported.")
        return ddf

    def preprocess(self, ddf):
        """Apply null-filling and custom preprocess functions to feature columns.

        Args:
            ddf (pl.LazyFrame): Input lazy frame.

        Returns:
            pl.LazyFrame: Lazy frame with preprocessed columns.
        """
        all_cols = self.label_cols + self.feature_cols[::-1]
        col_names = ddf.collect_schema().names()
        for col in all_cols:
            name = col["name"]
            fill_na = None
            if col["dtype"] in ["str", str]:
                fill_na = col.get("fill_na", "")
            elif col["dtype"] in ["int", int]:
                fill_na = col.get("fill_na", 0)
            elif col["dtype"] in ["float", float]:
                fill_na = col.get("fill_na", 0.0)
            col_exist = name in col_names
            if (fill_na is not None) and col_exist:
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
                )
            if (fill_na is not None) and (not col_exist):
                ddf = ddf.with_columns(pl.col(name).fill_null(fill_na))
            if col.get("type") == "sequence" and isinstance(ddf.collect_schema()[name], pl.List):
                # Convert list to "^" seperated string for unified preprocessing of parquet and csv formats
                ddf = ddf.with_columns(
                    pl.col(name).cast(pl.List(pl.String)).list.join("^")
                )
        active_cols = [col["name"] for col in all_cols if col.get("active") != False]
        ddf = ddf.select(active_cols)
        return ddf

    def fit(self, train_ddf, meta_ddf=None, min_categr_count=1, num_buckets=10,
            rebuild_dataset=True, **kwargs):
        """Fit preprocessors (tokenizers, normalizers, etc.) on the training data.

        Args:
            train_ddf (pl.LazyFrame): Training data lazy frame.
            meta_ddf (pl.LazyFrame, optional): Lazy frame concatenating the meta
                columns of train/valid/test. Used to build a global vocabulary
                for meta features (e.g. group_id) so that IDs stay consistent
                across all splits. Default: ``None`` (falls back to train_ddf).
            min_categr_count (int): Minimum frequency for categorical tokens.
            num_buckets (int): Number of buckets for quantile or hash bucketing.
            rebuild_dataset (bool): Whether to collect data for fitting.
            **kwargs: Additional keyword arguments.
        """
        self.rebuild_dataset = rebuild_dataset
        if self.rebuild_dataset:
            logging.info("Collecting dataset statistics...")
            stats_dict = self._collect_statistics(train_ddf, meta_ddf)
        for col in self.feature_cols:
            name = col["name"]
            if col["active"]:
                logging.info("Processing column: {}".format(col))
                col_stats = stats_dict.get(name) if self.rebuild_dataset else None
                if col["type"] == "meta": # e.g. set group_id in gAUC
                    self.fit_meta_col(col, col_stats)
                elif col["type"] == "numeric":
                    self.fit_numeric_col(col, col_stats)
                elif col["type"] == "embedding":
                    self.fit_embedding_col(col)
                elif col["type"] == "categorical":
                    col_input = col_stats
                    if col.get("category_processor") in ("quantile_bucket", "hash_bucket"):
                        col_input = (
                            train_ddf.select(name).collect().to_series().to_pandas() 
                            if self.rebuild_dataset else None
                        )
                    self.fit_categorical_col(col, col_input,
                                             min_categr_count=min_categr_count,
                                             num_buckets=num_buckets)
                elif col["type"] == "sequence":
                    self.fit_sequence_col(col, col_stats,
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
        logging.info("Setting up feature processor done.")

    def _collect_statistics(self, train_ddf, meta_ddf=None):
        """Compute per-column statistics via three lazy queries in one streaming pass.

        Args:
            train_ddf (pl.LazyFrame): Training data lazy frame.
            meta_ddf (pl.LazyFrame, optional): Lazy frame for the global meta vocab.

        Returns:
            dict: ``name -> stats`` where stats is one of
            ``{"value_counts": {token: count}}`` /
            ``{"max_len": int}`` /
            ``{"min": .., "max": .., "mean": .., "std": .., "count": ..}``.
        """
        cat_cols = [c["name"] for c in self.feature_cols
                    if c.get("active", True) and c["type"] == "categorical" and c.get("remap", True)]
        meta_cols = [c["name"] for c in self.feature_cols
                     if c.get("active", True) and c["type"] == "meta" and c.get("remap", True)]
        seq_cols = [c["name"] for c in self.feature_cols
                    if c.get("active", True) and c["type"] == "sequence" and c.get("remap", True)]
        num_cols = [c["name"] for c in self.feature_cols
                    if c.get("active", True) and c["type"] == "numeric" and "normalizer" in c]

        # exclude categorical columns that use quantile/hash bucket (kept on raw values)
        bucket_cols = {c["name"] for c in self.feature_cols
                       if c.get("category_processor") in ("quantile_bucket", "hash_bucket")}
        cat_cols = [c for c in cat_cols if c not in bucket_cols]
        feature_spec = {c["name"]: c for c in self.feature_cols}

        lazy_frames = []
        count_cols, maxlen_cols = [], []
        # Track per-column lazy query index for reading results
        count_indices = {}  # col_name -> index in lazy_frames

        if meta_ddf is None:
            meta_ddf = train_ddf
        for c in meta_cols + cat_cols:
            source_ddf = meta_ddf if c in meta_cols else train_ddf
            col_lf = (source_ddf.select([
                        pl.lit(c).alias("feature"),
                        pl.col(c).alias("value")
                      ])
                      .drop_nulls()
                      .group_by(["feature", "value"])
                      .len()
                      .sort(["feature", "len", "value"],
                            descending=[False, True, False]))
            count_indices[c] = len(lazy_frames)
            lazy_frames.append(col_lf)
            count_cols.append(c)
        for c in seq_cols:
            splitter = feature_spec[c].get("splitter", "^")
            col_lf = (train_ddf.select([
                        pl.lit(c).alias("feature"),
                        pl.col(c).str.split(splitter).alias("value")
                      ])
                      .explode("value")
                      .drop_nulls()
                      .group_by(["feature", "value"])
                      .len()
                      .sort(["feature", "len", "value"],
                            descending=[False, True, False]))
            count_indices[c] = len(lazy_frames)
            lazy_frames.append(col_lf)
            count_cols.append(c)

        # (2) max sequence length (not neccessary for columns with manually set max_len)
        maxlen_exprs = []
        for c in seq_cols:
            splitter = feature_spec[c].get("splitter", "^")
            max_len = feature_spec[c].get("max_len", 0)
            if max_len == 0:
                maxlen_exprs.append(
                    pl.col(c).str.split(splitter).list.len().max().alias(c))
        if maxlen_exprs:
            maxlen_idx = len(lazy_frames)
            lazy_frames.append(train_ddf.select(maxlen_exprs))
            maxlen_cols = [c for c in seq_cols
                           if feature_spec[c].get("max_len", 0) == 0]

        # (3) numeric statistics for normalizer reconstruction
        exprs = []
        for c in num_cols:
            col_s = pl.col(c).drop_nulls()
            exprs += [col_s.min().alias(f"{c}::min"),
                      col_s.max().alias(f"{c}::max"),
                      col_s.mean().alias(f"{c}::mean"),
                      col_s.std(ddof=0).alias(f"{c}::std"),
                      col_s.count().alias(f"{c}::count")]
        if num_cols:
            num_idx = len(lazy_frames)
            lazy_frames.append(train_ddf.select(exprs))

        results = pl.collect_all(lazy_frames, engine="streaming") if lazy_frames else []

        stats = {}
        for name in cat_cols:
            stats[name] = {"value_counts": {}}
        for name in meta_cols:
            stats[name] = {"value_counts": {}}
        for name in seq_cols:
            stats[name] = {"value_counts": {}, "max_len": 0}
        for name in num_cols:
            stats[name] = {}

        # Read back per-column count results (each preserves its own dtype)
        for c in count_cols:
            df_counts = results[count_indices[c]]
            for r in df_counts.iter_rows(named=True):
                token, cnt = r["value"], r["len"]
                stats[c]["value_counts"][token] = cnt

        # Read back max sequence lengths
        if maxlen_cols:
            df_maxlen = results[maxlen_idx]
            for name in maxlen_cols:
                stats[name]["max_len"] = int(df_maxlen[name][0])

        if num_cols:
            df_num = results[num_idx]
            for c in num_cols:
                stats[c] = {"min": df_num[f"{c}::min"][0],
                            "max": df_num[f"{c}::max"][0],
                            "mean": df_num[f"{c}::mean"][0],
                            "std": df_num[f"{c}::std"][0],
                            "count": df_num[f"{c}::count"][0]}
        return stats

    def fit_meta_col(self, col, stats_dict):
        """Fit a meta column (e.g. group_id) by registering its tokenizer.

        Args:
            col (dict): Column specification dict.
            stats_dict (dict or None): Global token counts from
                :meth:`_collect_statistics`, or None if ``rebuild_dataset`` is False.
        """
        name = col["name"]
        feature_type = col["type"]
        self.feature_map.features[name] = {"type": feature_type}
        if col.get("remap", True):
            tokenizer = Tokenizer(min_freq=1, remap=True)
            if stats_dict is not None:
                # stats dict: {"value_counts": {token: count}}
                word_counts = stats_dict.get("value_counts", stats_dict)
                tokenizer.build_vocab(word_counts)
            self.processor_dict[name + "::tokenizer"] = tokenizer

    def fit_numeric_col(self, col, stats_dict=None):
        """Fit a numeric column, optionally registering a normalizer.

        The normalizer is reconstructed from aggregate statistics (min/max/mean/std)
        computed by :meth:`_collect_statistics`.

        Args:
            col (dict): Column specification dict.
            stats_dict (dict or None): Statistics dict with keys
                ``min``/``max``/``mean``/``std``/``count``, or None if
                ``rebuild_dataset`` is False.
        """
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
            if self.rebuild_dataset and stats_dict is not None:
                normalizer = Normalizer(col["normalizer"])
                normalizer.fit_from_stats(min_value=stats_dict.get("min"),
                                          max_value=stats_dict.get("max"),
                                          mean=stats_dict.get("mean"),
                                          std=stats_dict.get("std"),
                                          count=stats_dict.get("count"))
                self.processor_dict[name + "::normalizer"] = normalizer

    def fit_embedding_col(self, col):
        """Fit an embedding column by recording its pretrain dimensions.

        Args:
            col (dict): Column specification dict.
        """
        name = col["name"]
        feature_type = col["type"]
        feature_source = col.get("source", "")
        self.feature_map.features[name] = {"source": feature_source,
                                           "type": feature_type}
        if "feature_encoder" in col:
            self.feature_map.features[name]["feature_encoder"] = col["feature_encoder"]
        if "embedding_dim" in col:
            self.feature_map.features[name]["embedding_dim"] = col["embedding_dim"]
        if "pretrain_dim" in col:
            self.feature_map.features[name]["pretrain_dim"] = col["pretrain_dim"]

    def fit_categorical_col(self, col, col_series=None, min_categr_count=1, num_buckets=10):
        """Fit a categorical column, building or loading its tokenizer vocab.

        Args:
            col (dict): Column specification dict.
            col_series (dict or pd.Series or None): Either the token counts from
                :meth:`_compute_statistics` (``{token: count}``) or raw values,
                or None if ``rebuild_dataset`` is False.
            min_categr_count (int): Minimum token frequency.
            num_buckets (int): Number of buckets for quantile/hash processors.
        """
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
                if isinstance(col_series, dict):  # counts from _collect_statistics
                    value_counts = col_series.get("value_counts", {})
                    tokenizer.build_vocab(value_counts)
            else:
                # rebuild_dataset=False: reconstruct from vocab_size
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
                    qtf.fit(col_series.values.reshape(-1,1))
                    boundaries = qtf.quantiles_[1:-1]
                    self.processor_dict[name + "::boundaries"] = boundaries
                self.feature_map.features[name]["vocab_size"] = num_buckets
            elif category_processor == "hash_bucket":
                num_buckets = col.get("num_buckets", num_buckets)
                self.feature_map.features[name]["vocab_size"] = num_buckets
                self.processor_dict[name + "::num_buckets"] = num_buckets
            else:
                raise NotImplementedError("category_processor={} not supported.".format(category_processor))

    def fit_sequence_col(self, col, stats_dict=None, min_categr_count=1):
        """Fit a sequence column, building its tokenizer with splitting and padding.

        Args:
            col (dict): Column specification dict.
            stats_dict (dict or None): Token counts from :meth:`_compute_statistics`
                (``{"value_counts": {...}, "max_len": int}``), or None if
                ``rebuild_dataset`` is False.
            min_categr_count (int): Minimum token frequency.
        """
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
            tokenizer.build_vocab(stats_dict["value_counts"])
            if tokenizer.max_len == 0:
                tokenizer.max_len = stats_dict.get("max_len", 0)
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

    def transform(self, batch):
        """Batched transform of raw feature columns into integer IDs or normalized values.

        Args:
            batch (dict): Mapping of column name to a list/array of values, as
                provided by ``datasets.map``.

        Returns:
            dict: Mapping of column name to transformed values.
        """
        for feature, feature_spec in self.feature_map.features.items():
            if feature not in batch:
                continue
            feature_type = feature_spec["type"]
            col_series = batch[feature]
            if feature_type == "meta":
                tokenizer = self.processor_dict.get(feature + "::tokenizer")
                if tokenizer is not None:
                    batch[feature] = np.array(
                        tokenizer.encode_category(col_series), dtype=np.int64)
            elif feature_type == "numeric":
                normalizer = self.processor_dict.get(feature + "::normalizer")
                if normalizer is not None:
                    batch[feature] = np.array(
                        normalizer.transform(np.array(col_series)), dtype=np.float64)
            elif feature_type == "categorical":
                category_processor = feature_spec.get("category_processor")
                if category_processor is None:
                    batch[feature] = np.array(
                        self.processor_dict.get(feature + "::tokenizer")
                        .encode_category(col_series), dtype=np.int64)
                elif category_processor == "numeric_bucket":
                    raise NotImplementedError
                elif category_processor == "hash_bucket":
                    raise NotImplementedError
            elif feature_type == "sequence":
                batch[feature] = np.array(
                    self.processor_dict.get(feature + "::tokenizer")
                    .encode_sequence(col_series), dtype=np.int64)
            elif feature_type == "embedding":
                continue
            else:
                raise NotImplementedError
        return batch

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
        """Serialize the feature processor to a pickle file.

        Args:
            pickle_file (str): Destination pickle path.
        """
        logging.info("Pickle feature_encode: " + pickle_file)
        pickle.dump(self, open(pickle_file, "wb"))

    def save_vocab(self, vocab_file):
        """Save categorical and sequence vocabularies to a JSON file.

        Args:
            vocab_file (str): Destination JSON path.
        """
        logging.info("Save feature_vocab to json: " + vocab_file)
        vocab = dict()
        for feature, spec in self.feature_map.features.items():
            if spec["type"] in ["categorical", "sequence"]:
                vocab[feature] = OrderedDict(
                    sorted(self.processor_dict[feature + "::tokenizer"].vocab.items(), key=lambda x:x[1]))
        with open(vocab_file, "w") as fd:
            fd.write(json.dumps(vocab, indent=4))

    def copy_from(self, src_col):
        """Return a Polars expression that copies another column verbatim.

        Args:
            src_col (str): Name of the source column to copy.

        Returns:
            pl.Expr: Polars column expression.
        """
        return pl.col(src_col)