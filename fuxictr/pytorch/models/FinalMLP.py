# =========================================================================
# Copyright (C) 2022 FuxiCTR Authors. All rights reserved.
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

import torch
import torch.nn as nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingDictLayer, MLP_Layer


class FinalMLP(BaseModel):
    def __init__(self, feature_map, model_id='FinalMLP', gpu=-1, task='binary_classification', learning_rate=1e-3,
                 embedding_dim=10, dnn1_hidden_units=[64, 64, 64], dnn2_hidden_units=[64, 64, 64],
                 dnn1_activations="ReLU", dnn2_activations="ReLU", dnn1_dropout=0., dnn2_dropout=0.,
                 dnn1_batch_norm=False, dnn2_batch_norm=False, user_feature_selection=True,
                 selection_hidden_units=[64], bilinear_group=1, embedding_regularizer=None, net_regularizer=None,
                 **kwargs):
        super(FinalMLP, self).__init__(feature_map, model_id=model_id, gpu=gpu, embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer, **kwargs)
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.dnn1 = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                              output_dim=None,
                              hidden_units=dnn1_hidden_units,
                              hidden_activations=dnn1_activations,
                              output_activation=None,
                              dropout_rates=dnn1_dropout,
                              batch_norm=dnn1_batch_norm)
        self.dnn2 = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                              output_dim=None,
                              hidden_units=dnn2_hidden_units,
                              hidden_activations=dnn2_activations,
                              output_activation=None,
                              dropout_rates=dnn2_dropout,
                              batch_norm=dnn2_batch_norm)
        self.user_fields = sum(1 if feature_spec["source"] == "user" else 0 \
                               for _, feature_spec in feature_map.feature_specs.items())
        self.item_fields = sum(1 if feature_spec["source"] == "item" else 0 \
                               for _, feature_spec in feature_map.feature_specs.items())
        self.selection = FeatureSelection(feature_map.num_fields, self.user_fields, self.item_fields,
                                         embedding_dim, user_feature_selection, selection_hidden_units)
        self.fusion = GroupBilinearAggregation(dnn1_hidden_units[-1],
                                               dnn2_hidden_units[-1],
                                               bilinear_group)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feat_emb_dict = self.embedding_layer(X)
        if self.user_fields > 0:
            user_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="user")
        else:
            user_emb = None
        if self.item_fields > 0:
            item_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="item")
        else:
            item_emb = None
        feature_emb = self.embedding_layer.dict2tensor(feat_emb_dict)
        feature1, feature2 = self.selection(feature_emb, user_emb, item_emb)
        y_pred = self.fusion(self.dnn1(feature1), self.dnn2(feature2))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class FeatureSelection(nn.Module):
    def __init__(self, num_fields, user_fields, item_fields, embedding_dim,
                 user_feature_selection, selection_hidden_units):
        super(FeatureSelection, self).__init__()
        self.use_feature_selection = user_feature_selection
        if self.use_feature_selection:
            self.user_fields = user_fields
            self.item_fields = item_fields
        if user_fields > 0:
            self.user_gate = MLP_Layer(input_dim=embedding_dim * user_fields,
                                       output_dim=num_fields,
                                       hidden_units=selection_hidden_units,
                                       hidden_activations="ReLU",
                                       output_activation="Sigmoid")
        else:
            self.user_gate = nn.Parameter(torch.ones(num_fields))

        if item_fields > 0:
            self.item_gate = MLP_Layer(input_dim=embedding_dim * item_fields,
                                       output_dim=num_fields,
                                       hidden_units=selection_hidden_units,
                                       hidden_activations="ReLU",
                                       output_activation="Sigmoid")
        else:
            self.item_gate = nn.Parameter(torch.ones(num_fields))

    def forward(self, feature_emb, user_emb, item_emb):
        if self.use_feature_selection:
            if self.user_fields > 0:
                gate_u = self.user_gate(user_emb.flatten(start_dim=1))
            else:
                gate_u = self.user_gate.sigmoid()

            if self.item_fields > 0:
                gate_i = self.item_gate(user_emb.flatten(start_dim=1))
            else:
                gate_i = self.item_gate.sigmoid()
            feature1 = (feature_emb.tanh() * gate_u.unsqueeze(-1)).flatten(start_dim=1)
            feature2 = (feature_emb.tanh() * gate_i.unsqueeze(-1)).flatten(start_dim=1)
            return feature1, feature2
        else:
            feature = feature_emb.flatten(start_dim=1)
            return feature, feature


class GroupBilinearAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, group=1):
        super(GroupBilinearAggregation, self).__init__()
        self.group = group
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.w_x = nn.Linear(x_dim, 1)
        self.w_y = nn.Linear(y_dim, 1, bias=False)
        if self.group > 0:
            assert x_dim % group == 0 and y_dim % group == 0,\
            "The last hidden dim must be divisible by the number of bilinear groups"
            self.w_xy = nn.Linear(x_dim * y_dim // group, 1 , bias=False)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        if self.group > 0:
            group_x = x.view(-1, self.group, self.x_dim // self.group)
            group_y = y.view(-1, self.group, self.y_dim // self.group)
            xy = torch.einsum("bhm,bhn->bhmn", group_x, group_y).flatten(start_dim=1)
            output += self.w_xy(xy)
        return output