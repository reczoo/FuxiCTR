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
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer, InnerProductLayer


class DLRM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DLRM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 top_mlp_units=[64, 64, 64],
                 bottom_mlp_units=[64, 64, 64],
                 top_mlp_activations="ReLU",
                 bottom_mlp_activations="ReLU",
                 top_mlp_dropout=0,
                 bottom_mlp_dropout=0,
                 interaction_op="dot", # ["dot", "cat"]
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DLRM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.feature_map = feature_map
        self.dense_feats = [feat for feat, feature_spec in feature_map.feature_specs.items() \
                            if feature_spec["type"] == "numeric"]
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim,
                                              not_required_feature_columns=self.dense_feats)
        if len(self.dense_feats) > 0:
            n_fields = feature_map.num_fields - len(self.dense_feats) + 1 # add processed dense feature
            self.bottom_mlp = MLP_Layer(input_dim=len(self.dense_feats),
                                        output_dim=embedding_dim,
                                        hidden_units=bottom_mlp_units,
                                        hidden_activations=bottom_mlp_activations,
                                        output_activation=bottom_mlp_activations,
                                        dropout_rates=bottom_mlp_dropout, 
                                        batch_norm=batch_norm)
        else:
            n_fields = feature_map.num_fields
        self.interaction_op = interaction_op
        if self.interaction_op == "dot":
            self.interact = InnerProductLayer(num_fields=n_fields, output="inner_product")
            top_input_dim = (n_fields * (n_fields - 1)) // 2 + embedding_dim * int(len(self.dense_feats) > 0)
        elif self.interaction_op == "cat":
            top_input_dim = n_fields * embedding_dim
        else:
            raise ValueError("interaction_op={} not supported.".format(self.interaction_op))
        self.top_mlp = MLP_Layer(input_dim=top_input_dim,
                                 output_dim=1, 
                                 hidden_units=top_mlp_units,
                                 hidden_activations=top_mlp_activations,
                                 output_activation=self.get_output_activation(task),
                                 dropout_rates=top_mlp_dropout,
                                 batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feat_emb = self.embedding_layer(X)
        if len(self.dense_feats) > 0:
            feature_indexes = [self.feature_map.feature_specs[f]["index"] for f in self.dense_feats]
            dense_x = torch.cat([X[:, idx].float().view(-1, 1) for idx in feature_indexes], dim=-1)
            dense_emb = self.bottom_mlp(dense_x)
            feat_emb = torch.cat([feat_emb, dense_emb.unsqueeze(1)], dim=1)
        if self.interaction_op == "dot":
            interact_out = self.interact(feat_emb)
        else:
            interact_out = feat_emb.flatten(start_dim=1)
        if self.interaction_op == "dot" and len(self.dense_feats) > 0:
            interact_out = torch.cat([interact_out, dense_emb], dim=-1)
        y_pred = self.top_mlp(interact_out)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict