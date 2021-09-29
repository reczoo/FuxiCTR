# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .embedding import EmbeddingLayer
from .interaction import InnerProductLayer, InnerProductLayer_v2
from itertools import combinations


class LR_Layer(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(LR_Layer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.final_activation = final_activation
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = EmbeddingLayer(feature_map, 1)

    def forward(self, X):
        embed_weights = self.embedding_layer(X)
        output = torch.stack(embed_weights).sum(dim=0)
        if self.bias is not None:
            output += self.bias
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


class FM_Layer(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer, self).__init__()
        self.inner_product_layer = InnerProductLayer(output="sum")
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation

    def forward(self, X, feature_emb_list):
        lr_out = self.lr_layer(X)
        dot_out = self.inner_product_layer(feature_emb_list)
        output = dot_out + lr_out
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


class FM_Layer_v2(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer_v2, self).__init__()
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields, output="sum")
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation

    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X)
        dot_sum = self.inner_product_layer(feature_emb)
        output = dot_sum + lr_out
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output

