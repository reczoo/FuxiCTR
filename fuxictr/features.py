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


from collections import OrderedDict
import io
import os
import logging
import json


class FeatureMap(object):
    def __init__(self, dataset_id, data_dir):
        self.data_dir = data_dir # must keep to be used in embedding layer for pretrained emb
        self.dataset_id = dataset_id
        self.num_fields = 0
        self.total_features = 0
        self.input_length = 0
        self.features = OrderedDict()
        self.labels = []
        self.column_index = dict()
        self.group_id = None
        self.default_emb_dim = None

    def load(self, json_file, params):
        logging.info("Load feature_map from json: " + json_file)
        with io.open(json_file, "r", encoding="utf-8") as fd:
            feature_map = json.load(fd) #, object_pairs_hook=OrderedDict
        if feature_map["dataset_id"] != self.dataset_id:
            raise RuntimeError("dataset_id={} does not match feature_map!".format(self.dataset_id))
        self.labels = feature_map.get("labels", [])
        self.total_features = feature_map.get("total_features", 0)
        self.input_length = feature_map.get("input_length", 0)
        self.group_id = params.get("group_id", None)
        self.default_emb_dim = params.get("embedding_dim", None)
        self.features = OrderedDict((k, v) for x in feature_map["features"] for k, v in x.items())
        self.num_fields = self.get_num_fields()
        if params.get("use_features", None):
            self.features = OrderedDict((x, self.features[x]) for x in params["use_features"])
        if params.get("feature_specs", None):
            self.update_feature_specs(params["feature_specs"])
        self.set_column_index()

    def update_feature_specs(self, feature_specs):
        for col in feature_specs:
            namelist = col["name"]
            if type(namelist) != list:
                namelist = [namelist]
            for name in namelist:
                for k, v in col.items():
                    if k != "name":
                        self.features[name][k] = v

    def save(self, json_file):
        logging.info("Save feature_map to json: " + json_file)
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        feature_map = OrderedDict()
        feature_map["dataset_id"] = self.dataset_id
        feature_map["num_fields"] = self.num_fields
        feature_map["total_features"] = self.total_features
        feature_map["input_length"] = self.input_length
        feature_map["labels"] = self.labels
        feature_map["features"] = [{k: v} for k, v in self.features.items()]
        with open(json_file, "w") as fd:
            json.dump(feature_map, fd, indent=4)

    def get_num_fields(self, feature_source=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        num_fields = 0
        for feature, feature_spec in self.features.items():
            if feature_spec["type"] == "meta":
                continue
            if len(feature_source) == 0 or feature_spec.get("source") in feature_source:
                num_fields += 1
        return num_fields

    def sum_emb_out_dim(self, feature_source=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        total_dim = 0
        for feature, feature_spec in self.features.items():
            if feature_spec["type"] == "meta":
                continue
            if len(feature_source) == 0 or feature_spec.get("source") in feature_source:
                total_dim += feature_spec.get("emb_output_dim",
                                              feature_spec.get("embedding_dim", 
                                                               self.default_emb_dim))
        return total_dim

    def set_column_index(self):
        logging.info("Set column index...")
        idx = 0
        for feature, feature_spec in self.features.items():
            if "max_len" in feature_spec:
                col_indexes = [i + idx for i in range(feature_spec["max_len"])]
                self.column_index[feature] = col_indexes
                idx += feature_spec["max_len"]
            else:
                self.column_index[feature] = idx
                idx += 1
        self.input_length = idx
        for label in self.labels:
            self.column_index[label] = idx
            idx += 1

    def get_column_index(self, feature):
        if feature not in self.column_index:
            self.set_column_index()
        return self.column_index[feature]

        
