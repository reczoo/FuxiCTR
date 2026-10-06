# =========================================================================
# Copyright (C) 2026. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import runpy
import sys
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest
import yaml

from fuxictr import preprocess, utils
from fuxictr.datasets.criteo import CustomizedFeatureProcessor


class DatasetBuilt(Exception):
    pass


@pytest.fixture
def run_experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_expid.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(utils, "set_logger", lambda params: None)
    for name, attrs in {
        "fuxictr.pytorch.dataloaders": {"RankDataLoader": None},
        "fuxictr.pytorch.torch_utils": {"seed_everything": lambda seed: None},
        "model_zoo": {},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "model_config.yaml").write_text(yaml.safe_dump({
        "test_experiment": {"dataset_id": "test_dataset", "seed": 2026}
    }))
    monkeypatch.setattr(sys, "argv", [str(script), "--config", str(config_dir),
                                      "--expid", "test_experiment"])
    captured = {}

    def build_dataset(processor, **params):
        captured["processor"] = processor
        captured["values"] = processor.preprocess(pl.LazyFrame({
            "value": [1.0, 3.0, 10.0], "label": [0.0, 1.0, 0.0]
        })).collect()["value"].to_list()
        raise DatasetBuilt

    monkeypatch.setattr(preprocess, "build_dataset", build_dataset)

    def run(customized_feature_processor=None):
        column = {"name": "value", "dtype": "float", "type": "numeric"}
        params = {"data_root": str(tmp_path), "feature_cols": [column],
                  "label_col": {"name": "label", "dtype": "float"}}
        if customized_feature_processor is not None:
            params["customized_feature_processor"] = customized_feature_processor
            column["preprocess"] = "convert_to_bucket"
        (config_dir / "dataset_config.yaml").write_text(yaml.safe_dump({
            "test_dataset": params
        }))
        runpy.run_path(str(script), run_name="__main__")

    return run, captured


def test_default_feature_processor(run_experiment):
    run, captured = run_experiment
    with pytest.raises(DatasetBuilt):
        run()
    assert type(captured["processor"]) is preprocess.FeatureProcessor
    assert captured["values"] == [1.0, 3.0, 10.0]


def test_custom_feature_processor_from_dataset_config(run_experiment):
    run, captured = run_experiment
    with pytest.raises(DatasetBuilt):
        run("fuxictr.datasets.criteo.CustomizedFeatureProcessor")
    assert type(captured["processor"]) is CustomizedFeatureProcessor
    assert captured["values"] == [1, 1, 5]


@pytest.mark.parametrize("processor_path,error", [
    ("nonexistent_processor.CustomizedFeatureProcessor", ModuleNotFoundError),
    ("fuxictr.datasets.criteo.NonexistentProcessor", AttributeError),
])
def test_invalid_custom_processor_fails_before_building_dataset(
        run_experiment, processor_path, error):
    run, captured = run_experiment
    with pytest.raises(error):
        run(processor_path)
    assert captured == {}
