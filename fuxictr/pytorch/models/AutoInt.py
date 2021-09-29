# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import DNN_Layer, EmbeddingLayer, MultiHeadSelfAttention, LR_Layer

class AutoInt(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AutoInt", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=False,
                 use_wide=False,
                 use_residual=True,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(AutoInt, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=False) \
                        if use_wide else None
        self.dnn = DNN_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case no DNN used
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else num_heads * attention_dim,
                                    attention_dim=attention_dim, 
                                    num_heads=num_heads, 
                                    dropout_rate=net_dropout, 
                                    use_residual=use_residual, 
                                    use_scale=use_scale, 
                                    layer_norm=layer_norm,
                                    align_to="output") 
             for i in range(attention_layers)])
        self.fc = nn.Linear(feature_map.num_fields * attention_dim * num_heads, 1)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X)
        feature_emb_tensor = torch.stack(feature_emb_list, dim=1)
        attention_out = self.self_attention(feature_emb_tensor)
        attention_out = torch.flatten(attention_out, start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            concate_feature_emb = torch.cat(feature_emb_list, dim=1)
            y_pred += self.dnn(concate_feature_emb)
        if self.lr_layer is not None:
            y_pred += self.lr_layer(X)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict

