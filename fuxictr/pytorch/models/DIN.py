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

""" This model implements the paper "Zhou et al., Deep Interest Network for 
    Click-Through Rate Prediction, KDD'2018".
    [PDF] https://arxiv.org/pdf/1706.06978.pdf
    [Code] https://github.com/zhougr1993/DeepInterestNetwork
"""

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingDictLayer, MLP_Layer
from ..layers.activation import Dice


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
                 din_target_field="item_id",
                 din_sequence_field="click_history",
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DIN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        for query, history in zip(self.din_target_field, self.din_sequence_field):
            if query not in feature_map.feature_specs or history not in feature_map.feature_specs:
                raise ValueError("din_target_field or din_sequence_field is not correct!")
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList([DINAttentionLayer(embedding_dim,
                                                                 attention_units=attention_units,
                                                                 hidden_activations=attention_activations,
                                                                 final_activation=attention_final_activation,
                                                                 dice_alpha=dice_alpha,
                                                                 dropout_rate=net_dropout,
                                                                 batch_norm=batch_norm)
                                               for _ in range(len(self.din_sequence_field))])
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
        for idx, (din_target_field, din_sequence_field) \
            in enumerate(zip(self.din_target_field, self.din_sequence_field)):
            target_item = feature_emb_dict[din_target_field]
            history_sequence_emb = feature_emb_dict[din_sequence_field]
            pooled_history_emb = self.attention_layers[idx](target_item, history_sequence_emb)
            feature_emb_dict[din_sequence_field] = pooled_history_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class DINAttentionLayer(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 attention_units=[32], 
                 hidden_activations="ReLU",
                 final_activation=None,
                 dice_alpha=0.,
                 dropout_rate=0,
                 batch_norm=False):
        super(DINAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
            hidden_activations = [Dice(units, alpha=dice_alpha) for units in attention_units]
        self.attention_layer = MLP_Layer(input_dim=4 * embedding_dim,
                                         output_dim=1,
                                         hidden_units=attention_units,
                                         hidden_activations=hidden_activations,
                                         final_activation=final_activation,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm, 
                                         use_bias=True)

    def forward(self, target_item, history_sequence):
        # target_item: b x emd
        # history_sequence: b x len x emb
        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([target_item, history_sequence, target_item - history_sequence, 
                                     target_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        # mask = history_sequence.sum(dim=-1) != 0
        # attention_weight = attention_weight * mask.float()
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1) # mask by all zeros
        return output

