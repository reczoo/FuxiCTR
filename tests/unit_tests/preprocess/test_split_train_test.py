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

import sys
sys.path.append("../../")

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from fuxictr.preprocess.build_dataset import split_train_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int) -> pl.DataFrame:
    """Create a simple DataFrame with an `id` column [0, n) for tracking rows."""
    return pl.DataFrame({"id": list(range(n)), "value": [float(i) * 0.1 for i in range(n)]})


def row_ids(df: pl.DataFrame) -> list:
    return df["id"].to_list()


# ---------------------------------------------------------------------------
# No split
# ---------------------------------------------------------------------------

class TestNoSplit:
    def test_returns_original_train_when_sizes_are_zero(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=0, test_size=0)
        assert_frame_equal(df,train)
        assert valid is None
        assert test is None

    def test_pre_existing_valid_and_test_passed_through(self):
        train_df = make_df(80)
        valid_df = make_df(10)
        test_df = make_df(10)
        train, valid, test = split_train_test(
            train_df, valid_ddf=valid_df, test_ddf=test_df, valid_size=0, test_size=0
        )
        # No further splitting should occur; original DataFrames returned as-is.
        assert_frame_equal(train_df,train)
        assert_frame_equal(valid_df,valid)
        assert_frame_equal(test_df,test)


# ---------------------------------------------------------------------------
# Sequential split — absolute integer sizes
# ---------------------------------------------------------------------------

class TestSequentialAbsoluteSize:
    def test_valid_only_absolute(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=20, test_size=0)
        assert_frame_equal(df.slice(0,80),train)
        assert_frame_equal(df.slice(80,20),valid)        
        assert test is None

    def test_test_only_absolute(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=0, test_size=20)
        assert_frame_equal(df.slice(0,80),train)
        assert_frame_equal(df.slice(80,20),test)        
        assert valid is None

    def test_both_absolute(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=10, test_size=20)
        assert len(train) == 70
        assert len(valid) == 10
        assert len(test) == 20

    def test_sequential_train_ids_are_first_rows(self):
        """Sequential split must take the last rows for test/valid."""
        df = make_df(10)
        train, valid, test = split_train_test(df, valid_size=2, test_size=3)
        # test = rows 7,8,9; valid = rows 4,5,6; train = rows 0,1,2,3
        assert row_ids(test) == [7, 8, 9]
        assert row_ids(valid) == [5, 6]
        assert row_ids(train) == [0, 1, 2, 3, 4]

    def test_total_row_count_preserved(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=15, test_size=25)
        assert len(train) + len(valid) + len(test) == 100


# ---------------------------------------------------------------------------
# Sequential split — fractional sizes
# ---------------------------------------------------------------------------

class TestSequentialFractionalSize:
    def test_valid_fraction(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=0.2, test_size=0)
        assert len(valid) == 20
        assert len(train) == 80

    def test_test_fraction(self):
        df = make_df(100)
        train, valid, test = split_train_test(df, valid_size=0, test_size=0.3)
        assert len(test) == 30
        assert len(train) == 70

    def test_both_fractions(self):
        df = make_df(200)
        train, valid, test = split_train_test(df, valid_size=0.1, test_size=0.2)
        # test = 0.2 * 200 = 40; valid = 0.1 * 200 = 20; train = 200 - 40 - 20 = 140
        assert len(test) == 40
        assert len(valid) == 20
        assert len(train) == 140

    def test_fraction_floor_division(self):
        """Fractional sizes are int()-truncated, not rounded."""
        df = make_df(10)
        train, valid, test = split_train_test(df, valid_size=0.15, test_size=0)
        # int(10 * 0.15) == 1
        assert len(valid) == 1
        assert len(train) == 9


# ---------------------------------------------------------------------------
# Random split
# ---------------------------------------------------------------------------

class TestRandomSplit:
    def test_row_counts_same_as_sequential(self):
        df = make_df(100)
        train, valid, test = split_train_test(
            df, valid_size=20, test_size=10, split_type="random"
        )
        assert len(train) == 70
        assert len(valid) == 20
        assert len(test) == 10

    def test_no_duplicate_ids_across_splits(self):
        df = make_df(100)
        train, valid, test = split_train_test(
            df, valid_size=20, test_size=10, split_type="random"
        )
        all_ids = row_ids(train) + row_ids(valid) + row_ids(test)
        assert len(all_ids) == len(set(all_ids)), "Duplicate rows found across splits"

    def test_all_original_ids_present(self):
        df = make_df(100)
        train, valid, test = split_train_test(
            df, valid_size=20, test_size=10, split_type="random"
        )
        all_ids = sorted(row_ids(train) + row_ids(valid) + row_ids(test))
        assert all_ids == list(range(100))

    def test_random_splits_differ_from_sequential(self):
        """With a fixed seed, random order should (almost certainly) differ from sequential."""
        np.random.seed(42)
        df = make_df(100)
        train_rand, _, _ = split_train_test(
            df, valid_size=20, test_size=0, split_type="random"
        )
        train_seq, _, _ = split_train_test(
            df, valid_size=20, test_size=0, split_type="sequential"
        )
        assert row_ids(train_rand) != row_ids(train_seq)


# ---------------------------------------------------------------------------
# LazyFrame input
# ---------------------------------------------------------------------------

class TestLazyFrameInput:
    def test_lazyframe_is_collected_and_split(self):
        lazy_df = make_df(50).lazy()
        train, valid, test = split_train_test(lazy_df, valid_size=10, test_size=5)
        assert isinstance(train, pl.DataFrame)
        assert len(train) == 35
        assert len(valid) == 10
        assert len(test) == 5

    def test_lazyframe_no_split_returns_dataframe(self):
        lazy_df = make_df(30).lazy()
        train, valid, test = split_train_test(lazy_df, valid_size=0, test_size=0)
        assert isinstance(train, pl.DataFrame)
        assert len(train) == 30
