# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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

import os
import glob
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from collections import Counter

from fuxictr.preprocess import Tokenizer
from fuxictr.preprocess.normalizer import Normalizer
from fuxictr.preprocess.feature_processor import FeatureProcessor


def _make_processor(tmpdir):
    """Build a FeatureProcessor with pre-fitted tokenizers/normalizer.

    processor_dict is populated directly to avoid the full fit pipeline; only
    transform() is exercised here.
    """
    fp = FeatureProcessor(
        feature_cols=[
            {"name": "user_id", "type": "categorical", "dtype": "str"},
            {"name": "click_history", "type": "sequence", "dtype": "str",
             "splitter": "|", "max_len": 5, "padding": "post"},
            {"name": "age", "type": "numeric", "dtype": "float"},
            {"name": "group_id", "type": "meta", "dtype": "str"},
        ],
        label_col=[{"name": "label", "type": "float", "dtype": "float"}],
        dataset_id="test_transform",
        data_root=tmpdir,
    )
    fp.rebuild_dataset = True

    cat_tok = Tokenizer(min_freq=1, remap=True)
    cat_tok.build_vocab(Counter({"u0": 2, "u1": 2}))
    fp.processor_dict["user_id::tokenizer"] = cat_tok

    seq_tok = Tokenizer(min_freq=1, splitter="|", max_len=5,
                        padding="post", remap=True)
    seq_tok.build_vocab(Counter({"a": 3, "b": 2, "c": 1}))
    fp.processor_dict["click_history::tokenizer"] = seq_tok

    meta_tok = Tokenizer(min_freq=1, remap=True)
    meta_tok.build_vocab(Counter({"g1": 3, "g2": 1}))
    fp.processor_dict["group_id::tokenizer"] = meta_tok

    norm = Normalizer("MinMaxScaler")
    # Normalizer.fit 已移除（用户决策），改用 fit_from_stats 直接重建
    x = np.array([0., 10., 20.])
    norm.fit_from_stats(min_value=float(np.min(x)), max_value=float(np.max(x)),
                        mean=float(np.mean(x)), std=float(np.std(x)), count=int(len(x)))
    fp.processor_dict["age::normalizer"] = norm

    fp.feature_map.features = {
        "user_id": {"type": "categorical"},
        "click_history": {"type": "sequence", "max_len": 5},
        "age": {"type": "numeric"},
        "group_id": {"type": "meta"},
    }
    fp.feature_map.labels = ["label"]
    return fp


def _make_batch():
    """Return a batch dict (column -> list) as datasets.map would provide."""
    return {
        "user_id": ["u0", "u1", "u0", "u2"],          # u2 is OOV
        "click_history": ["a|b", "b|c", "a", ""],     # "" -> na -> pad
        "age": [0., 10., 20., 5.],
        "group_id": ["g1", "g2", "g1", "g3"],          # g3 is OOV
        "label": [1., 0., 1., 0.],
    }


class TestTransformBatch:
    """Unit tests for FeatureProcessor.transform as a datasets.map batch callback."""

    def test_transform_batch_encodes_all_types(self):
        tmpdir = tempfile.mkdtemp()
        try:
            fp = _make_processor(tmpdir)
            out = fp.transform(_make_batch())
            assert isinstance(out, dict)
            assert set(out.keys()) == set(_make_batch().keys())

            cat_tok = fp.processor_dict["user_id::tokenizer"]
            assert out["user_id"][0] == cat_tok.vocab["u0"]
            assert out["user_id"][1] == cat_tok.vocab["u1"]
            assert out["user_id"][3] == cat_tok.vocab["__OOV__"]

            seqs = out["click_history"]
            assert all(len(s) == 5 for s in seqs)
            seq_tok = fp.processor_dict["click_history::tokenizer"]
            assert seqs[0][0] == seq_tok.vocab["a"]
            assert seqs[0][1] == seq_tok.vocab["b"]
            assert seqs[2][0] == seq_tok.vocab["a"]

            assert out["age"][0] == pytest.approx(0.0)
            assert out["age"][2] == pytest.approx(1.0)
            assert out["age"][3] == pytest.approx(0.25)

            meta_tok = fp.processor_dict["group_id::tokenizer"]
            assert out["group_id"][0] == meta_tok.vocab["g1"]
            assert out["group_id"][3] == meta_tok.vocab["__OOV__"]

            assert list(out["label"]) == [1., 0., 1., 0.]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_transform_batch_does_not_mutate_processor_dict(self):
        tmpdir = tempfile.mkdtemp()
        try:
            fp = _make_processor(tmpdir)
            vocab_before = dict(fp.processor_dict["user_id::tokenizer"].vocab)
            fp.transform(_make_batch())
            assert fp.processor_dict["user_id::tokenizer"].vocab == vocab_before
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestBuildDatasetTransform:
    """End-to-end tests for build_dataset.transform (map + write parquet)."""

    def test_transform_default_writes_single_file(self):
        from fuxictr.preprocess.build_dataset import transform
        import polars as pl
        tmpdir = tempfile.mkdtemp()
        try:
            fp = _make_processor(tmpdir)
            os.makedirs(fp.data_dir, exist_ok=True)
            lf = pl.from_pandas(pd.DataFrame(_make_batch())).lazy()
            transform(fp, lf, "train")  # block_size=0 -> single file
            path = os.path.join(fp.data_dir, "train.parquet")
            assert os.path.exists(path)
            assert not os.path.isdir(os.path.join(fp.data_dir, "train"))
            read_df = pd.read_parquet(path)
            assert len(read_df) == 4
            cat_tok = fp.processor_dict["user_id::tokenizer"]
            assert int(read_df["user_id"].iloc[0]) == cat_tok.vocab["u0"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_transform_block_size_writes_parts(self):
        from fuxictr.preprocess.build_dataset import transform
        import polars as pl
        tmpdir = tempfile.mkdtemp()
        try:
            fp = _make_processor(tmpdir)
            os.makedirs(fp.data_dir, exist_ok=True)
            df = pd.concat([pd.DataFrame(_make_batch())] * 10, ignore_index=True)  # 40 rows
            lf = pl.from_pandas(df).lazy()
            transform(fp, lf, "train", block_size=15)
            out_dir = os.path.join(fp.data_dir, "train")
            parts = sorted(glob.glob(os.path.join(out_dir, "part-*.parquet")))
            assert len(parts) == 3  # 15 + 15 + 10
            read_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            assert len(read_df) == 40
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
