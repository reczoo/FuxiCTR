# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, EmbeddingDictLayer, DNN_Layer, Dice


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
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
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
        self.dnn = DNN_Layer(input_dim=feature_map.num_fields * embedding_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=self.get_final_activation(task), 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

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
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
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
        self.attention_layer = DNN_Layer(input_dim=4 * embedding_dim,
                                         output_dim=1,
                                         hidden_units=attention_units,
                                         hidden_activations=hidden_activations,
                                         final_activation=final_activation,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm, 
                                         use_bias=True)

    def forward(self, query_item, history_sequence):
        # query_item: b x emd
        # history_sequence: b x len x emb
        seq_len = history_sequence.size(1)
        query_item = query_item.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([query_item, history_sequence, query_item - history_sequence, 
                                     query_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1) # mask by all zeros
        return output

