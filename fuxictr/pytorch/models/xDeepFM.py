# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import LR_Layer, EmbeddingLayer, DNN_Layer


class xDeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="xDeepFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU",
                 cin_layer_units=[16, 16, 16], 
                 embedding_dropout=0,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(xDeepFM, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)     
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = DNN_Layer(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only CIN used
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=False)
        self.cin = CIN(feature_map.num_fields, cin_layer_units, output_dim=1)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X) # list of b x embedding_dim
        lr_logit = self.lr_layer(X)
        cin_logit = self.cin(feature_emb_list)
        if self.dnn is not None:
            concate_feature_emb = torch.cat(feature_emb_list, dim=1) # b x input_dim
            dnn_logit = self.dnn(concate_feature_emb)
            y_pred = lr_logit + cin_logit + dnn_logit # LR + CIN + DNN
        else:
            y_pred = lr_logit + cin_logit # only LR + CIN
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict


class CIN(nn.Module):
    def __init__(self, num_fields, cin_layer_units, output_dim=1):
        super(CIN, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, feature_emb_list):
        pooling_outputs = []
        X_0 = torch.stack(feature_emb_list, dim=1) # b x field_size x emb_size
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        output = self.fc(concate_vec)
        return output