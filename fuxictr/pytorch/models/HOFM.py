# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import LR_Layer, EmbeddingLayer_v2, InnerProductLayer_v2
from itertools import combinations


class HOFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="HOFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 order=3,
                 embedding_dim=10,
                 reuse_embedding=False,
                 embedding_dropout=0,
                 regularizer=None, 
                 **kwargs):
        super(HOFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs)
        self.order = order
        assert order >= 2, "order >= 2 is required in HOFM!"
        self.reuse_embedding = reuse_embedding
        if reuse_embedding:
            assert isinstance(embedding_dim, int), "embedding_dim should be an integer when reuse_embedding=True."
            self.embedding_layer = EmbeddingLayer_v2(feature_map, 
                                                     embedding_dim, 
                                                     embedding_dropout)
        else:
            if not isinstance(embedding_dim, list):
                embedding_dim = [embedding_dim] * (order - 1)
            self.embedding_layers = nn.ModuleList([EmbeddingLayer_v2(feature_map, 
                                                                     embedding_dim[i], 
                                                                     embedding_dropout) \
                                                   for i in range(order - 1)])
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields)
        self.lr_layer = LR_Layer(feature_map, use_bias=True)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
        self.field_conjunction_dict = dict()
        for order_i in range(3, self.order + 1):
            order_i_conjunction = zip(*list(combinations(range(feature_map.num_fields), order_i)))
            for k, field_index in enumerate(order_i_conjunction):
                self.field_conjunction_dict[(order_i, k)] = torch.LongTensor(field_index)

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        y_pred = self.lr_layer(X)
        if self.reuse_embedding:
            feature_emb = self.embedding_layer(X)
        for i in range(2, self.order + 1):
            order_i_out = self.high_order_interaction(feature_emb if self.reuse_embedding \
                                                      else self.embedding_layers[i - 2](X), order_i=i)
            y_pred += order_i_out
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def high_order_interaction(self, feature_emb, order_i):
        if order_i == 2:
            interaction_out = self.inner_product_layer(feature_emb)
        elif order_i > 2:
            index = self.field_conjunction_dict[(order_i, 0)].to(self.device)
            hadamard_product = torch.index_select(feature_emb, 1, index)
            for k in range(1, order_i):
                index = self.field_conjunction_dict[(order_i, k)].to(self.device)
                hadamard_product = hadamard_product * torch.index_select(feature_emb, 1, index)
            interaction_out = hadamard_product.sum((1, 2)).view(-1, 1)
        return interaction_out
