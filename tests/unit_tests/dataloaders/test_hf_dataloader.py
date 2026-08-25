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
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import directly from the module file to avoid triggering the
# dataloaders/__init__.py -> rank_dataloader -> torch import chain.
# The module itself imports torch at top level (inherits DataLoader),
# so it can only be loaded when torch is available.
import importlib.util

hf_mod = None
if HAS_TORCH:
    _mod_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "fuxictr", "pytorch", "dataloaders", "hf_dataloader.py"
    ))
    _spec = importlib.util.spec_from_file_location("hf_dataloader", _mod_path)
    hf_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(hf_mod)


def _make_parquet(tmpdir, name, n_rows=100):
    """Write a small parquet file with categorical + sequence + label columns."""
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n_rows)],
        "item_id": [f"i{i % 10}" for i in range(n_rows)],
        "click_history": ["|".join(str(j) for j in range(i % 5 + 1))
                           for i in range(n_rows)],
        "label": [float(i % 2) for i in range(n_rows)],
    })
    path = os.path.join(tmpdir, f"{name}.parquet")
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def _make_feature_map():
    """Build a minimal FeatureMap-compatible mock.

    Follows the real FuxiCTR semantics: ``features`` holds only the feature
    specs (label is NOT a feature), and ``labels`` is a separate list.
    """
    from fuxictr.features import FeatureMap

    fm = FeatureMap("test_hf", "./tmp_test_hf")
    fm.labels = ["label"]
    fm.features = {
        "user_id": {"type": "categorical", "vocab_size": 100},
        "item_id": {"type": "categorical", "vocab_size": 10},
        "click_history": {"type": "sequence", "max_len": 5},
    }
    fm.set_column_index()
    return fm


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestHFDataLoaderLoadData:
    """Test load_data()."""

    def test_load_single_parquet(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 50)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=8, shuffle=False, num_workers=0)
            assert dl.num_samples == 50
            assert dl.num_blocks == 1  # single parquet file
            row = dl.dataset[0]
            assert "user_id" in row
            assert "label" in row
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_parquet_dir(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "part_0", 30)
            _make_parquet(tmpdir, "part_1", 20)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, tmpdir, batch_size=8, shuffle=False,
                                     num_workers=0)
            assert dl.num_samples == 50
            assert dl.num_blocks == 2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_getitem_returns_expected_value(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 5)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=8, shuffle=False, num_workers=0)
            row = dl.dataset[2]
            assert str(row["user_id"]) == "u2"  # tensor -> str
            assert row["label"].item() == 0.0  # i=2 -> label 0
            assert str(row["item_id"]) == "i2"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_missing_dir_raises(self):
        tmpdir = tempfile.mkdtemp()
        try:
            fm = _make_feature_map()
            with pytest.raises(AssertionError):
                hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "empty"),
                                    batch_size=8, shuffle=False, num_workers=0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestHFDataLoader:

    def test_is_data_loader_subclass(self):
        import torch.utils.data
        assert issubclass(hf_mod.HFDataLoader, torch.utils.data.DataLoader)
        assert not hf_mod.HFDataLoader.__bases__[0] is object

    def test_dataloader_basic_iteration(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 100)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=32, shuffle=False, num_workers=0)
            assert dl.num_samples == 100
            assert dl.num_batches == 4  # ceil(100/32)
            assert len(dl) == 4

            n_batches = 0
            for batch in dl:
                n_batches += 1
                assert "user_id" in batch
                assert "label" in batch
                # Numeric label -> tensor; string features -> list (with_format torch)
                assert len(batch["user_id"]) <= 32
                assert batch["label"].shape[0] <= 32
            assert n_batches == 4
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dataloader_shuffle_changes_order(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 50)
            fm = _make_feature_map()
            dl_no = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                        batch_size=50, shuffle=False, num_workers=0)
            dl_sh = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                        batch_size=50, shuffle=True, num_workers=0)
            batch_no = next(iter(dl_no))
            batch_sh = next(iter(dl_sh))
            users_no = [str(u) for u in batch_no["user_id"]]
            users_sh = [str(u) for u in batch_sh["user_id"]]
            assert sorted(users_no) == sorted(users_sh)
            assert users_no != users_sh
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dataloader_multi_parquet_dir(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "part_0", 60)
            _make_parquet(tmpdir, "part_1", 40)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, tmpdir, batch_size=32, shuffle=False,
                                     num_workers=0)
            assert dl.num_samples == 100
            assert dl.num_batches == 4
            assert dl.num_blocks == 2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dataloader_last_batch_partial(self):
        """Last batch has fewer rows than batch_size."""
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 70)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=32, shuffle=False, num_workers=0)
            assert dl.num_batches == 3  # ceil(70/32) = 3
            batches = list(dl)
            assert len(batches) == 3
            assert len(batches[0]["user_id"]) == 32
            assert len(batches[1]["user_id"]) == 32
            assert len(batches[2]["user_id"]) == 6  # 70 - 64 = 6
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestHFDataLoaderStreaming:
    """Tests for streaming mode (streaming=True)."""

    def test_streaming_num_samples(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 50)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=8, shuffle=False, num_workers=0,
                                     streaming=True)
            assert dl.num_samples == 50
            assert dl.num_blocks == 1
            assert dl.streaming is True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_streaming_iteration(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 100)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=32, shuffle=False, num_workers=0,
                                     streaming=True)
            assert dl.num_samples == 100
            assert dl.num_batches == 4  # ceil(100/32)
            assert len(dl) == 4
            n_batches = 0
            for batch in dl:
                n_batches += 1
                assert "user_id" in batch
                assert "label" in batch
                assert batch["label"].shape[0] <= 32
            assert n_batches == 4
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_streaming_multi_parquet(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "part_0", 60)
            _make_parquet(tmpdir, "part_1", 40)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, tmpdir, batch_size=32, shuffle=False,
                                     num_workers=0, streaming=True)
            assert dl.num_samples == 100
            assert dl.num_batches == 4
            assert dl.num_blocks == 2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_streaming_last_batch_partial(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 70)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=32, shuffle=False, num_workers=0,
                                     streaming=True)
            assert dl.num_batches == 3  # ceil(70/32)
            batches = list(dl)
            assert len(batches) == 3
            assert len(batches[0]["user_id"]) == 32
            assert len(batches[1]["user_id"]) == 32
            assert len(batches[2]["user_id"]) == 6  # 70 - 64
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_streaming_shuffle_no_loss(self):
        tmpdir = tempfile.mkdtemp()
        try:
            _make_parquet(tmpdir, "train", 50)
            fm = _make_feature_map()
            dl = hf_mod.HFDataLoader(fm, os.path.join(tmpdir, "train.parquet"),
                                     batch_size=10, shuffle=True, num_workers=0,
                                     streaming=True, buffer_size=20)
            users = []
            for batch in dl:
                users.extend(str(u) for u in batch["user_id"])
            assert len(users) == 50
            assert set(users) == {f"u{i}" for i in range(50)}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])