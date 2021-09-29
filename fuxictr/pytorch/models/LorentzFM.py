# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v2, InnerProductLayer_v2
from itertools import combinations

class LorentzFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="LorentzFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 embedding_dropout=0,
                 regularizer=None, 
                 **kwargs):
        super(LorentzFM, self).__init__(feature_map, 
                                        model_id=model_id, 
                                        gpu=gpu, 
                                        embedding_regularizer=regularizer, 
                                        net_regularizer=regularizer,
                                        **kwargs)
        self.embedding_layer = EmbeddingLayer_v2(feature_map, 
                                                 embedding_dim, 
                                                 embedding_dropout)
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields, output="dot_vector")
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X) # bs x field x dim
        inner_product = self.inner_product_layer(feature_emb) # bs x (field x (field - 1) / 2)
        zeroth_components = self.get_zeroth_components(feature_emb) # batch * field
        y_pred = self.triangle_pooling(inner_product, zeroth_components) 
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict

    def get_zeroth_components(self, feature_emb):
        '''
        compute the 0th component
        '''
        sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
        zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
        return zeroth_components # batch * field

    def triangle_pooling(self, inner_product, zeroth_components):
        '''
        T(u,v) = (1 - <u, v>L - u0 - v0) / (u0 * v0)
               = (1 + u0 * v0 - inner_product - u0 - v0) / (u0 * v0)
               = 1 + (1 - inner_product - u0 - v0) / (u0 * v0)
        '''
        num_fields = zeroth_components.size(1)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        u0, v0 = zeroth_components[:, p], zeroth_components[:, q]  # batch * (f(f-1)/2)
        score_tensor = 1 + torch.div(1 - inner_product - u0 - v0, u0 * v0) # batch * (f(f-1)/2)
        output = torch.sum(score_tensor, dim=1) # batch * 1
        return output