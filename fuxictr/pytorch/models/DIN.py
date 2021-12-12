# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

""" This model implements the paper: Zhou et al., Deep Interest Network for 
    Click-Through Rate Prediction, KDD'2018.
    [PDF] https://arxiv.org/pdf/1706.06978.pdf
    [Code] https://github.com/zhougr1993/DeepInterestNetwork
"""

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingDictLayer, MLP_Layer, DINAttentionLayer


class DIN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN", 
                 gpu=-1, 
                 task="binary_classification",
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_units=[64],
                 attention_activations="Dice",
                 attention_final_activation=None,
                 dice_alpha=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_query_field="item_id",
                 din_history_field="click_history",
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DIN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_query_field, list):
            din_query_field = [din_query_field]
        self.din_query_field = din_query_field
        if not isinstance(din_history_field, list):
            din_history_field = [din_history_field]
        self.din_history_field = din_history_field
        for query, history in zip(self.din_query_field, self.din_history_field):
            if query not in feature_map.feature_specs or history not in feature_map.feature_specs:
                raise ValueError("din_query_field or din_history_field is not correct!")
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList([DINAttentionLayer(embedding_dim,
                                                                 attention_units=attention_units,
                                                                 hidden_activations=attention_activations,
                                                                 final_activation=attention_final_activation,
                                                                 dice_alpha=dice_alpha,
                                                                 dropout_rate=net_dropout,
                                                                 batch_norm=batch_norm)
                                               for _ in range(len(self.din_history_field))])
        self.dnn = MLP_Layer(input_dim=feature_map.num_fields * embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=self.get_final_activation(task), 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.apply(self.init_weights)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for idx, (din_query_field, din_history_field) \
            in enumerate(zip(self.din_query_field, self.din_history_field)):
            item_emb = feature_emb_dict[din_query_field]
            history_sequence_emb = feature_emb_dict[din_history_field]
            pooled_history_emb = self.attention_layers[idx](item_emb, history_sequence_emb)
            feature_emb_dict[din_history_field] = pooled_history_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict




