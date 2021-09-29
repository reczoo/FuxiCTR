# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import numpy as np
from torch import nn
import torch
from ...pytorch.utils import set_activation

class DNN_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 final_activation=None, 
                 dropout_rates=[], 
                 batch_norm=False, 
                 use_bias=True):
        super(DNN_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation is not None:
            dense_layers.append(set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.dnn(inputs)


class FGCNN_Layer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """
    def __init__(self, 
                 num_fields, 
                 embedding_dim,
                 channels=[3], 
                 kernel_heights=[3], 
                 pooling_sizes=[2],
                 recombined_channels=[2],
                 activation="Tanh",
                 batch_norm=True):
        super(FGCNN_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        conv_list = []
        recombine_list = []
        self.channels = [1] + channels # input channel = 1
        input_height = num_fields
        for i in range(1, len(self.channels)):
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            pooling_size = pooling_sizes[i - 1]
            recombined_channel = recombined_channels[i - 1]
            conv_layer = [nn.Conv2d(in_channel, out_channel, 
                                    kernel_size=(kernel_height, 1), 
                                    padding=(int((kernel_height - 1) / 2), 0))] \
                       + ([nn.BatchNorm2d(out_channel)] if batch_norm else []) \
                       + [set_activation(activation),
                          nn.MaxPool2d((pooling_size, 1), padding=(input_height % pooling_size, 0))]
            conv_list.append(nn.Sequential(*conv_layer))
            input_height = int(np.ceil(input_height / pooling_size))
            input_dim =  input_height * embedding_dim * out_channel
            output_dim = input_height * embedding_dim * recombined_channel
            recombine_layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                            set_activation(activation))
            recombine_list.append(recombine_layer)
        self.conv_layers = nn.ModuleList(conv_list)
        self.recombine_layers = nn.ModuleList(recombine_list)

    def forward(self, X):
        conv_out = X
        new_feature_list = []
        for i in range(len(self.channels) - 1):
            conv_out = self.conv_layers[i](conv_out)
            flatten_out = torch.flatten(conv_out, start_dim=1)
            recombine_out = self.recombine_layers[i](flatten_out)
            new_feature_list.append(recombine_out.reshape(X.size(0), -1, self.embedding_dim))
        new_feature_emb = torch.cat(new_feature_list, dim=1)
        return new_feature_emb

