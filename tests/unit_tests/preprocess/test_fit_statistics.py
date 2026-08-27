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

"""Unit tests for the fit-phase statistics refactor.

Covers:
- ``FeatureProcessor._collect_statistics``: token counts / max_len / numeric
  stats computed by three lazy queries collected in one streaming pass.
- ``Tokenizer.build_vocab``: accepts a plain ``dict`` (polars-derived) with
  deterministic ordering.
- ``Normalizer.fit_from_stats``: reconstructs StandardScaler / MinMaxScaler
  from aggregate statistics, transforming identically to a real fit.
- ``fit_meta_col`` / ``fit_categorical_col`` / ``fit_sequence_col`` /
  ``fit_numeric_col``: consume the new stats structures.
- quantile_bucket / hash_bucket paths keep their original behavior.
"""

import numpy as np
import polars as pl
import pandas as pd
import pytest
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fuxictr.preprocess import Tokenizer
from fuxictr.preprocess.normalizer import Normalizer
from fuxictr.preprocess.feature_processor import FeatureProcessor


def _make_fp(feature_cols, data_root="./tmp_test_fit_stats"):
    return FeatureProcessor(
        feature_cols=feature_cols,
        label_col=[{"name": "label", "type": "float", "dtype": "float"}],
        dataset_id="test_ds",
        data_root=data_root,
    )


SAMPLE_COLS = [
    {"name": "user_id", "type": "categorical", "dtype": "str", "active": True},
    {"name": "item_id", "type": "categorical", "dtype": "str", "active": True},
    {"name": "click_seq", "type": "sequence", "dtype": "str", "active": True},
    {"name": "buy_seq", "type": "sequence", "dtype": "str", "active": True},
    {"name": "num_f", "type": "numeric", "dtype": "float", "normalizer": "StandardScaler", "active": True},
    {"name": "group_id", "type": "meta", "dtype": "str", "active": True},
]


def _sample_df():
    return pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u1", "u3", "u1", "u2"],
            "item_id": ["i1^a", "i1", "i2", "i1", "i3", "i2"],  # ^ inside a categorical value
            "click_seq": ["1^2", "2", "1^2^3", "3", "1", "2^3"],
            "buy_seq": ["1", "2", "3", "4", "5", "6"],
            "num_f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group_id": ["g1", "g2", "g1", "g3", "g1", "g2"],
        }
    )


class TestComputeStatistics:
    def test_counts_match_column_wise_oracle(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        df = _sample_df()
        stats = fp._collect_statistics(df.lazy())

        # oracle: count per column independently
        oracle = {}
        for c in ["user_id", "item_id", "group_id"]:
            vc = df[c].value_counts()
            oracle[c] = dict(zip(vc[c].to_list(), vc["count"].to_list()))
        for c in ["click_seq", "buy_seq"]:
            exp = df.select(pl.col(c).str.split("^").alias("v")).explode("v")["v"]
            vc = exp.value_counts()
            oracle[c] = dict(zip(vc["v"].to_list(), vc["count"].to_list()))

        for c, expect in oracle.items():
            assert stats[c]["value_counts"] == expect, f"mismatch on {c}"
        # categorical value containing the splitter is preserved intact
        assert stats["item_id"]["value_counts"]["i1^a"] == 1

    def test_max_len_detected(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        stats = fp._collect_statistics(_sample_df().lazy())
        assert stats["click_seq"]["max_len"] == 3  # "1^2^3"
        assert stats["buy_seq"]["max_len"] == 1

    def test_numeric_stats_are_ddof0(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        stats = fp._collect_statistics(_sample_df().lazy())
        n = stats["num_f"]
        assert n["min"] == 1.0 and n["max"] == 6.0
        assert n["mean"] == 3.5
        assert n["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), rel=1e-6)
        assert n["count"] == 6

    def test_meta_counts_come_from_meta_ddf(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        df = _sample_df()
        # meta frame: only the group_id column, one extra token not in train
        meta_df = pl.concat([df.select("group_id"),
                             pl.DataFrame({"group_id": ["g_new"]})])
        stats = fp._collect_statistics(df.lazy(), meta_df.lazy())
        # g_new only appears in the meta frame -> global vocab
        assert stats["group_id"]["value_counts"]["g_new"] == 1

    def test_quantile_hash_columns_excluded_from_counts(self):
        cols = SAMPLE_COLS + [
            {"name": "qcol", "type": "categorical", "dtype": "float",
             "category_processor": "quantile_bucket", "num_buckets": 4, "active": True},
            {"name": "hcol", "type": "categorical", "dtype": "str",
             "category_processor": "hash_bucket", "num_buckets": 8, "active": True},
        ]
        fp = _make_fp(cols)
        fp.rebuild_dataset = True
        df = _sample_df().with_columns([
            pl.Series("qcol", [1.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
            pl.Series("hcol", ["a", "b", "a", "c", "b", "a"]),
        ])
        stats = fp._collect_statistics(df.lazy())
        assert "qcol" not in stats
        assert "hcol" not in stats


class TestBuildVocabDictInput:
    def test_build_vocab_accepts_plain_dict(self):
        t = Tokenizer(min_freq=1, remap=True)
        t.build_vocab({"b": 1, "a": 3, "c": 2})
        # sorted by (-count, token): a(3), c(2), b(1) -> ids 1,2,3
        assert t.vocab["a"] == 1
        assert t.vocab["c"] == 2
        assert t.vocab["b"] == 3
        assert t.vocab["__PAD__"] == 0
        assert t.vocab["__OOV__"] == 4

    def test_build_vocab_dict_matches_counter(self):
        t1 = Tokenizer(min_freq=1, remap=True)
        t1.build_vocab(Counter({"a": 3, "b": 2, "c": 1}))
        t2 = Tokenizer(min_freq=1, remap=True)
        t2.build_vocab({"a": 3, "b": 2, "c": 1})
        assert t1.vocab == t2.vocab

    def test_build_vocab_empty_dict_keeps_reserved(self):
        t = Tokenizer(min_freq=1, remap=True)
        t.build_vocab({})
        assert set(t.vocab.keys()) == {"__PAD__", "__OOV__"}

    def test_min_freq_applied_to_dict_input(self):
        t = Tokenizer(min_freq=3, remap=True)
        t.build_vocab({"a": 5, "b": 2, "c": 1})
        assert "a" in t.vocab
        assert "b" not in t.vocab
        assert "c" not in t.vocab


class TestNormalizerFitFromStats:
    def test_standard_scaler_rebuild_equals_fit(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 3.0, 2000)
        s = pl.Series("x", x)
        stats = {"min": s.min(), "max": s.max(), "mean": s.mean(),
                 "std": s.std(ddof=0), "count": s.count()}
        n1 = Normalizer("StandardScaler")
        n1.fit_from_stats(min_value=stats["min"], max_value=stats["max"],
                          mean=stats["mean"], std=stats["std"], count=stats["count"])
        oracle = StandardScaler().fit(x.reshape(-1, 1)).transform(x.reshape(-1, 1)).flatten()
        assert np.allclose(n1.transform(x), oracle, rtol=1e-8, atol=1e-10)

    def test_min_max_scaler_rebuild_equals_fit(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-2.0, 8.0, 2000)
        s = pl.Series("x", x)
        stats = {"min": s.min(), "max": s.max(), "mean": s.mean(),
                 "std": s.std(ddof=0), "count": s.count()}
        n1 = Normalizer("MinMaxScaler")
        n1.fit_from_stats(min_value=stats["min"], max_value=stats["max"],
                          mean=stats["mean"], std=stats["std"], count=stats["count"])
        oracle = MinMaxScaler().fit(x.reshape(-1, 1)).transform(x.reshape(-1, 1)).flatten()
        assert np.allclose(n1.transform(x), oracle, rtol=1e-8, atol=1e-10)

    def test_standard_scaler_constant_column(self):
        x = np.full(200, 2.0)
        n1 = Normalizer("StandardScaler")
        n1.fit_from_stats(min_value=2.0, max_value=2.0, mean=2.0, std=0.0, count=200)
        oracle = StandardScaler().fit(x.reshape(-1, 1)).transform(x.reshape(-1, 1)).flatten()
        assert np.allclose(n1.transform(x), oracle)

    def test_min_max_scaler_constant_column(self):
        x = np.full(100, 3.5)
        n1 = Normalizer("MinMaxScaler")
        n1.fit_from_stats(min_value=3.5, max_value=3.5, mean=3.5, std=0.0, count=100)
        oracle = MinMaxScaler().fit(x.reshape(-1, 1)).transform(x.reshape(-1, 1)).flatten()
        assert np.allclose(n1.transform(x), oracle)


class TestFitColsWithStats:
    def test_fit_meta_col_consumes_stats_dict(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        fp.fit_meta_col({"name": "group_id", "type": "meta", "remap": True},
                        {"value_counts": {"g1": 3, "g2": 2, "g3": 1}})
        t = fp.processor_dict["group_id::tokenizer"]
        assert t.vocab["g1"] == 1 and t.vocab["g2"] == 2 and t.vocab["g3"] == 3

    def test_fit_categorical_col_consumes_counts_dict(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        # 保留 value_counts 契约：_collect_statistics 产出 {"value_counts": {...}}
        fp.fit_categorical_col({"name": "user_id", "type": "categorical"},
                               {"value_counts": {"u1": 3, "u2": 2, "u3": 1}})
        t = fp.processor_dict["user_id::tokenizer"]
        assert t.vocab["u1"] == 1
        assert fp.feature_map.features["user_id"]["vocab_size"] == 5  # __PAD__, 3 tokens, __OOV__

    def test_fit_sequence_col_uses_max_len_stat(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        # 保留 value_counts 契约：_collect_statistics 对 seq 产出 {"value_counts", "max_len"}
        fp.fit_sequence_col({"name": "click_seq", "type": "sequence"},
                            {"value_counts": {"1": 3, "2": 3, "3": 1}, "max_len": 3})
        t = fp.processor_dict["click_seq::tokenizer"]
        assert t.max_len == 3
        assert "1" in t.vocab

    def test_fit_numeric_col_rebuilds_from_stats(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        fp.fit_numeric_col(
            {"name": "num_f", "type": "numeric", "normalizer": "StandardScaler"},
            {"min": 1.0, "max": 6.0, "mean": 3.5,
             "std": float(np.std([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])), "count": 6})
        norm = fp.processor_dict["num_f::normalizer"]
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expect = StandardScaler().fit(x.reshape(-1, 1)).transform(x.reshape(-1, 1)).flatten()
        assert np.allclose(norm.transform(x), expect)

    def test_quantile_bucket_kept_on_raw_values(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        # 保留原真值方案（用户决策）：quantile_bucket 仍对真实值拟合，
        # 代码路径原样未动（qtf.fit(col_series.values)）。
        # 注：真实数据流中 col_series 是 pd.Series，.values 为 1D，sklearn>=1.6
        # 的 QuantileTransformer.fit 要求 2D —— 这是既有版本兼容问题，不在本次改造范围；
        # 此处传 2D DataFrame 仅验证 boundaries/vocab_size 保留逻辑未被重构破坏。
        fp.fit_categorical_col(
            {"name": "qcol", "type": "categorical", "category_processor": "quantile_bucket",
             "num_buckets": 5},
            pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]}))
        assert "qcol::boundaries" in fp.processor_dict
        assert fp.feature_map.features["qcol"]["vocab_size"] == 5

    def test_hash_bucket_needs_no_data(self):
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = True
        fp.fit_categorical_col(
            {"name": "hcol", "type": "categorical", "category_processor": "hash_bucket",
             "num_buckets": 8},
            None)
        assert fp.processor_dict["hcol::num_buckets"] == 8
        assert fp.feature_map.features["hcol"]["vocab_size"] == 8

    def test_meta_col_none_series_keeps_empty_vocab(self):
        """rebuild_dataset=False path: col_series=None -> no vocab built."""
        fp = _make_fp(SAMPLE_COLS)
        fp.rebuild_dataset = False
        fp.fit_meta_col({"name": "group_id", "type": "meta", "remap": True}, None)
        assert len(fp.processor_dict["group_id::tokenizer"].vocab) == 0


if __name__ == "__main__":
    pytest.main([__file__])