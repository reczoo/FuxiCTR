# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch
from .base_model import BaseModel
from ..layers import EmbeddingLayer
from ...pytorch.utils import set_activation

class CCPM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="CCPM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 channels=[4, 4, 2],
                 kernel_heights=[6, 5, 3],
                 activation="Tanh",
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 embedding_dropout=0,
                 **kwargs):
        super(CCPM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs) 
        self.embedding_layer = EmbeddingLayer(feature_map, 
                                              embedding_dim, 
                                              embedding_dropout=embedding_dropout)
        self.conv_layer = CCPM_ConvLayer(feature_map.num_fields, 
                                         channels=channels, 
                                         kernel_heights=kernel_heights, 
                                         activation=activation)
        conv_out_dim = 3 * embedding_dim * channels[-1] # 3 is k-max-pooling size of the last layer
        self.fc = nn.Linear(conv_out_dim, 1)
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
        conv_in = torch.unsqueeze(feature_emb_tensor, 1) # shape (bs, 1, field, emb)
        conv_out = self.conv_layer(conv_in)
        flatten_out = torch.flatten(conv_out, start_dim=1)
        y_pred = self.fc(flatten_out)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict


class CCPM_ConvLayer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """
    def __init__(self, num_fields, channels=[3], kernel_heights=[3], activation="Tanh"):
        super(CCPM_ConvLayer, self).__init__()
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        elif len(kernel_heights) != len(channels):
            raise ValueError("channels={} and kernel_heights={} should have the same length."\
                             .format(channels, kernel_heights))
        module_list = []
        self.channels = [1] + channels
        layers = len(kernel_heights)
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i - 1]
            out_channels = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            module_list.append(nn.ZeroPad2d((0, 0, kernel_height - 1, kernel_height - 1)))
            module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, 1)))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(KMaxPooling(k, dim=2))
            module_list.append(set_activation(activation))
        self.conv_layer = nn.Sequential(*module_list)

    def forward(self, X):
        return self.conv_layer(X)


class KMaxPooling(nn.Module):
    def __init__(self, k, dim):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, X):
        index = X.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        output = X.gather(self.dim, index)
        return output