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
sys.path.insert(0, "../../")

import pandas as pd
import pytest
from fuxictr.preprocess.tokenizer import Tokenizer


class TestEncodeMeta:
    """Regression tests for issue #164.

    Lazy tokenization of meta-type features must keep a consistent global
    vocabulary across blocks: no crash on the first call, and new words seen
    in later blocks get IDs that continue after the initial vocabulary.
    """

    def test_encode_meta_builds_vocab_on_first_call(self):
        """First call must not crash (previously: dict passed to build_vocab
        which calls .most_common() -> AttributeError)."""
        tok = Tokenizer(min_freq=1, remap=True)
        s = pd.Series(["a", "b", "a", "c", "d", "a"])
        out = tok.encode_meta(s)
        assert sorted(out) == [1, 1, 1, 2, 3, 4]  # a=1 (3x), others 2-4
        assert len(tok.vocab) == 6                # 4 words + PAD + OOV
        assert tok.vocab["__PAD__"] == 0
        assert tok.vocab["__OOV__"] == len(tok.vocab) - 1

    def test_encode_meta_new_words_get_contiguous_ids(self):
        """Words first seen in a later block must get IDs that continue the
        initial vocabulary (no collision with existing IDs or __OOV__)."""
        tok = Tokenizer(min_freq=1, remap=True)
        tok.encode_meta(pd.Series(["a", "b", "a", "c"]))        # block 1
        out2 = tok.encode_meta(pd.Series(["d", "a", "e"]))     # block 2

        # d < e alphabetically, so d gets the lower ID (deterministic)
        assert list(out2) == [5, 1, 6]
        real_ids = sorted(v for k, v in tok.vocab.items()
                          if k not in ("__PAD__", "__OOV__"))
        assert real_ids == [1, 2, 3, 5, 6]   # contiguous after initial vocab
        assert "d" in tok.vocab and "e" in tok.vocab
        assert tok.vocab["__OOV__"] == 7     # still points past max id

    def test_encode_meta_new_words_added_to_vocab(self):
        """Meta features use incremental vocab update: unseen tokens in
        later blocks are added to the vocabulary (not mapped to OOV).
        This is by design for meta-type features like group_id."""
        tok = Tokenizer(min_freq=1, remap=True)
        tok.encode_meta(pd.Series(["a", "b", "a"]))  # vocab: a=1, b=2
        out = tok.encode_meta(pd.Series(["a", "z"]))  # z is new -> added
        assert list(out) == [1, 4]   # z gets next available ID (3 was __OOV__)
        assert "z" in tok.vocab
        assert tok.vocab["z"] == 4
        assert tok.vocab["__OOV__"] == 5  # OOV moved past z
