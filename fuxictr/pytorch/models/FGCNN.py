# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, InnerProductLayer, MLP_Layer
from fuxictr.pytorch.torch_utils import get_activation


class FGCNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FGCNN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 share_embedding=False,
                 channels=[14, 16, 18, 20],
                 kernel_heights=[7, 7, 7, 7],
                 pooling_sizes=[2, 2, 2, 2],
                 recombined_channels=[2, 2, 2, 2],
                 conv_activation="Tanh",
                 conv_batch_norm=True,
                 dnn_hidden_units=[4096, 2048, 1024, 512],
                 dnn_activations="ReLU",
                 dnn_batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 net_dropout=0,
                 **kwargs):
        super(FGCNN, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.share_embedding = share_embedding
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        if not self.share_embedding:
            self.fg_embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        channels, kernel_heights, pooling_sizes, recombined_channels \
            = self.validate_input(channels, 
                                  kernel_heights, 
                                  pooling_sizes, 
                                  recombined_channels)
        self.fgcnn_layer = FGCNN_Layer(num_fields, 
                                       embedding_dim,
                                       channels=channels, 
                                       kernel_heights=kernel_heights, 
                                       pooling_sizes=pooling_sizes,
                                       recombined_channels=recombined_channels,
                                       activation=conv_activation,
                                       batch_norm=conv_batch_norm)
        input_dim, total_features = self.compute_input_dim(embedding_dim, 
                                                           num_fields, 
                                                           channels, 
                                                           pooling_sizes, 
                                                           recombined_channels)
        self.inner_product_layer = InnerProductLayer(total_features, output="inner_product")
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.get_output_activation(task),
                             dropout_rates=net_dropout,
                             batch_norm=dnn_batch_norm)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def compute_input_dim(self, 
                          embedding_dim, 
                          num_fields, 
                          channels, 
                          pooling_sizes, 
                          recombined_channels):
        total_features = num_fields
        input_height = num_fields
        for i in range(len(channels)):
            input_height = int(np.ceil(input_height / pooling_sizes[i]))
            total_features += input_height * recombined_channels[i]
        input_dim = int(total_features * (total_features - 1) / 2) \
                  + total_features * embedding_dim
        return input_dim, total_features

    def validate_input(self, 
                       channels, 
                       kernel_heights, 
                       pooling_sizes, 
                       recombined_channels):
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        if not isinstance(pooling_sizes, list):
            pooling_sizes = [pooling_sizes] * len(channels)
        if not isinstance(recombined_channels, list):
            recombined_channels = [recombined_channels] * len(channels)
        if not (len(channels) == len(kernel_heights) == len(pooling_sizes) == len(recombined_channels)):
            raise ValueError("channels, kernel_heights, pooling_sizes, and recombined_channels \
                              should have the same length.")
        return channels, kernel_heights, pooling_sizes, recombined_channels

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        if not self.share_embedding:
            feature_emb2 = self.fg_embedding_layer(X)
        else:
            feature_emb2 = feature_emb
        conv_in = torch.unsqueeze(feature_emb2, 1) # shape (bs, 1, field, emb)
        new_feature_emb = self.fgcnn_layer(conv_in)
        combined_feature_emb = torch.cat([feature_emb, new_feature_emb], dim=1)
        inner_product_vec = self.inner_product_layer(combined_feature_emb)
        dense_input = torch.cat([combined_feature_emb.flatten(start_dim=1), inner_product_vec], dim=1)
        y_pred = self.dnn(dense_input)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


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
                       + [get_activation(activation),
                          nn.MaxPool2d((pooling_size, 1), padding=(input_height % pooling_size, 0))]
            conv_list.append(nn.Sequential(*conv_layer))
            input_height = int(np.ceil(input_height / pooling_size))
            input_dim =  input_height * embedding_dim * out_channel
            output_dim = input_height * embedding_dim * recombined_channel
            recombine_layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                            get_activation(activation))
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