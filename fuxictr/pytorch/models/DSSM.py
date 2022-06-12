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
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingDictLayer, MLP_Layer


class DSSM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DSSM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 user_tower_units=[64, 64, 64],
                 item_tower_units=[64, 64, 64],
                 user_tower_activations="ReLU",
                 item_tower_activations="ReLU",
                 user_tower_dropout=0, 
                 item_tower_dropout=0, 
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DSSM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        user_fields = sum(1 if feature_spec["source"] == "user" else 0 \
                          for _, feature_spec in feature_map.feature_specs.items())
        item_fields = sum(1 if feature_spec["source"] == "item" else 0 \
                          for _, feature_spec in feature_map.feature_specs.items())
        self.user_tower = MLP_Layer(input_dim=embedding_dim * user_fields,
                                    output_dim=user_tower_units[-1],
                                    hidden_units=user_tower_units[0:-1],
                                    hidden_activations=user_tower_activations,
                                    output_activation=None,
                                    dropout_rates=user_tower_dropout, 
                                    batch_norm=batch_norm)
        self.item_tower = MLP_Layer(input_dim=embedding_dim * item_fields,
                                    output_dim=item_tower_units[-1], 
                                    hidden_units=item_tower_units[0:-1],
                                    hidden_activations=item_tower_activations,
                                    output_activation=None,
                                    dropout_rates=item_tower_dropout, 
                                    batch_norm=batch_norm)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feat_emb_dict = self.embedding_layer(X)
        user_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="user")
        item_emb = self.embedding_layer.dict2tensor(feat_emb_dict, feature_source="item")
        user_out = self.user_tower(user_emb.flatten(start_dim=1))
        item_out = self.item_tower(item_emb.flatten(start_dim=1))
        y_pred = (user_out * item_out).sum(dim=-1, keepdim=True)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
