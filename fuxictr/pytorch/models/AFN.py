# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import LR_Layer, EmbeddingLayer, DNN_Layer

class AFN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AFN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 embedding_dropout=0,
                 ensemble_dnn=True,
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU",
                 dnn_dropout=0,
                 afn_hidden_units=[64, 64, 64], 
                 afn_activations="ReLU",
                 afn_dropout=0,
                 logarithmic_neurons=5,
                 batch_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AFN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.num_fields = feature_map.num_fields
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout)
        self.coefficient_W = nn.Linear(self.num_fields, logarithmic_neurons, bias=False)
        self.dense_layer = DNN_Layer(input_dim=embedding_dim * logarithmic_neurons,
                                     output_dim=1, 
                                     hidden_units=afn_hidden_units,
                                     hidden_activations=afn_activations,
                                     final_activation=None, 
                                     dropout_rates=afn_dropout, 
                                     batch_norm=batch_norm, 
                                     use_bias=True)
        self.log_batch_norm = nn.BatchNorm1d(self.num_fields)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.ensemble_dnn = ensemble_dnn
        if ensemble_dnn:
            self.embedding_layer2 = EmbeddingLayer(feature_map, 
                                                   embedding_dim, 
                                                   embedding_dropout)
            self.dnn = DNN_Layer(input_dim=embedding_dim * self.num_fields,
                                 output_dim=1, 
                                 hidden_units=dnn_hidden_units,
                                 hidden_activations=dnn_activations,
                                 final_activation=None, 
                                 dropout_rates=dnn_dropout, 
                                 batch_norm=batch_norm, 
                                 use_bias=True)
            self.fc = nn.Linear(2, 1)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X)
        feature_emb = torch.stack(feature_emb_list, dim=1)
        dnn_input = self.logarithmic_net(feature_emb)
        afn_out = self.dense_layer(dnn_input)
        if self.ensemble_dnn:
            feature_emb_list2 = self.embedding_layer2(X)
            concate_feature_emb = torch.cat(feature_emb_list2, dim=1)
            dnn_out = self.dnn(concate_feature_emb)
            y_pred = self.fc(torch.cat([afn_out, dnn_out], dim=-1))
        else:
            y_pred = afn_out

        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-5) # ReLU with min 1e-5 (better than 1e-7 suggested in paper)
        log_feature_emb = torch.log(feature_emb) # element-wise log 
        log_feature_emb = self.log_batch_norm(log_feature_emb) # batch_size * num_fields * embedding_dim 
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out) # element-wise exp
        cross_out = self.exp_batch_norm(cross_out)  # batch_size * logarithmic_neurons * embedding_dim
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out
        