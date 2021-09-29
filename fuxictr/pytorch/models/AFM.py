# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer, InnerProductLayer, LR_Layer

class AFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="AFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 embedding_dropout=0,
                 attention_dropout=[0, 0],
                 attention_dim=10,
                 use_attention=True,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(AFM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.use_attention = use_attention
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        self.elementwise_product_layer = InnerProductLayer(output='element_wise')
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=True)
        self.attention = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                       nn.ReLU(),
                                       nn.Linear(attention_dim, 1, bias=False),
                                       nn.Softmax(dim=1))
        self.weight_p = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout[0])
        self.dropout2 = nn.Dropout(attention_dropout[1])
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb_list = self.embedding_layer(X)
        elementwise_product = self.elementwise_product_layer(feature_emb_list) # bs x f(f-1)/2 x dim
        if self.use_attention:
            attention_weight = self.attention(elementwise_product)
            attention_weight = self.dropout1(attention_weight)
            attention_sum = torch.sum(attention_weight * elementwise_product, dim=1)
            attention_sum = self.dropout2(attention_sum)
            afm_out = self.weight_p(attention_sum)
        else:
            afm_out = torch.flatten(elementwise_product, start_dim=1).sum(dim=-1).unsqueeze(-1)
        y_pred = self.lr_layer(X) + afm_out
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict
