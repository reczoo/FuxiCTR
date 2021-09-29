# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer, FiGNN_Layer


class FiGNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FiGNN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 embedding_dropout=0,
                 gnn_layers=3,
                 batch_size=32,
                 use_residual=True,
                 use_gru=True,
                 reuse_graph_layer=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FiGNN, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.init_batch_size = batch_size
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        self.fignn = FiGNN_Layer(self.num_fields, 
                                 self.embedding_dim,
                                 gnn_layers=gnn_layers,
                                 reuse_graph_layer=reuse_graph_layer,
                                 use_gru=use_gru,
                                 use_residual=use_residual,
                                 device=self.device)
        self.fc = PredictionLayer(self.num_fields, embedding_dim)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X)
        feature_emb = torch.stack(feature_emb_list, dim=1)
        h_out = self.fignn(feature_emb)
        y_pred = self.fc(h_out)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {'y_pred': y_pred, 'loss': loss}
        return return_dict


class PredictionLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(PredictionLayer, self).__init__()
        self.mlp1 = nn.Linear(embedding_dim, 1, bias=False)
        self.mlp2 = nn.Sequential(nn.Linear(num_fields * embedding_dim, num_fields, bias=False),
                                  nn.Sigmoid())

    def forward(self, h):
        score = self.mlp1(h).squeeze(-1) # b x f
        weight = self.mlp2(h.flatten(start_dim=1)) # b x f
        logit = (weight * score).sum(dim=1).unsqueeze(-1)
        return logit
