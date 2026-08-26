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

import pandas as pd
import pytest
from collections import Counter

from fuxictr.preprocess import Tokenizer
from fuxictr.preprocess.feature_processor import FeatureProcessor


class TestUpdateVocab(object):

    def test_update_vocab_reuses_oov_slot_then_moves_oov(self):
        """New words occupy the old __OOV__ slot; __OOV__ moves to the end."""
        tknzr = Tokenizer(min_freq=1, remap=True)
        tknzr.build_vocab(Counter({"a": 3, "b": 2, "c": 1}))
        # build_vocab: a=1, b=2, c=3, __OOV__=4
        oov_idx = tknzr.vocab["__OOV__"]
        tknzr.update_vocab(["x", "y"])
        # x and y occupy old __OOV__=4 and 5; __OOV__ moves to 6
        assert tknzr.vocab["x"] == oov_idx
        assert tknzr.vocab["y"] == oov_idx + 1
        assert tknzr.vocab["__OOV__"] == oov_idx + 2
        assert len(set(tknzr.vocab.values())) == len(tknzr.vocab)  # no collisions

    def test_update_vocab_on_empty_vocab(self):
        """update_vocab on an empty vocab: __OOV__ defaults to 0, first word gets 0."""
        tknzr = Tokenizer(min_freq=1, remap=True)
        tknzr.update_vocab(["a", "b"])
        # __OOV__ not in vocab -> get("__OOV__", 0) == 0; a=0, b=1; __OOV__ moves to 2
        assert tknzr.vocab["a"] == 0
        assert tknzr.vocab["b"] == 1
        assert tknzr.vocab["__OOV__"] == 2

    def test_update_vocab_deterministic_order(self):
        """Unordered input yields the same ids as sorted input."""
        tknzr1 = Tokenizer(min_freq=1, remap=True)
        tknzr1.build_vocab(Counter({"a": 3, "b": 2, "c": 1}))
        tknzr2 = Tokenizer(min_freq=1, remap=True)
        tknzr2.build_vocab(Counter({"a": 3, "b": 2, "c": 1}))
        words = ["zebra", "apple", "mango"]
        tknzr1.update_vocab(list(reversed(words)))
        tknzr2.update_vocab(words)
        assert tknzr1.vocab["apple"] == tknzr2.vocab["apple"]
        assert tknzr1.vocab["mango"] == tknzr2.vocab["mango"]
        assert tknzr1.vocab["zebra"] == tknzr2.vocab["zebra"]


class TestFitMetaCol(object):

    def _make_processor(self):
        return FeatureProcessor(
            feature_cols=[
                {"name": "group_id", "type": "meta", "dtype": "str"},
                {"name": "user_id", "type": "categorical", "dtype": "str"},
            ],
            label_col=[{"name": "label", "type": "float", "dtype": "float"}],
            dataset_id="test_dataset",
            data_root="./tmp_test_data",
        )

    def test_fit_meta_col_builds_global_vocab_from_series(self):
        fp = self._make_processor()
        fp.rebuild_dataset = True
        series = pd.Series(["g1", "g2", "g1", "g3"])
        fp.fit_meta_col({"name": "group_id", "type": "meta", "remap": True}, series)
        tknzr = fp.processor_dict["group_id::tokenizer"]
        assert "g1" in tknzr.vocab
        assert "g2" in tknzr.vocab
        assert "g3" in tknzr.vocab
        assert "__OOV__" in tknzr.vocab
        assert tknzr.vocab["g1"] != tknzr.vocab["g2"]

    def test_fit_meta_col_none_series_keeps_empty_vocab(self):
        """rebuild_dataset=False path: col_series is None -> no vocab built."""
        fp = self._make_processor()
        fp.rebuild_dataset = False
        fp.fit_meta_col({"name": "group_id", "type": "meta", "remap": True}, None)
        tknzr = fp.processor_dict["group_id::tokenizer"]
        assert len(tknzr.vocab) == 0


if __name__ == "__main__":
    pytest.main([__file__])