# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from torch import nn
import torch
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v2, FGCNN_Layer, InnerProductLayer_v2, DNN_Layer


class FGCNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FGCNN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
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
                 embedding_dropout=0,
                 net_dropout=0,
                 **kwargs):
        super(FGCNN, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.share_embedding = share_embedding
        self.embedding_layer = EmbeddingLayer_v2(feature_map, 
                                                 embedding_dim, 
                                                 embedding_dropout)
        if not self.share_embedding:
            self.fg_embedding_layer = EmbeddingLayer_v2(feature_map, 
                                                        embedding_dim, 
                                                        embedding_dropout)
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
        self.inner_product_layer = InnerProductLayer_v2(total_features, output="dot_vector")
        self.dnn = DNN_Layer(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=self.get_final_activation(task),
                             dropout_rates=net_dropout,
                             batch_norm=dnn_batch_norm)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

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
            feature_emb2 = feature_emb.clone()
        conv_in = torch.unsqueeze(feature_emb2, 1) # shape (bs, 1, field, emb)
        new_feature_emb = self.fgcnn_layer(conv_in)
        combined_feature_emb = torch.cat([feature_emb, new_feature_emb], dim=1)
        inner_product_vec = self.inner_product_layer(combined_feature_emb)
        dense_input = torch.cat([combined_feature_emb.flatten(start_dim=1), inner_product_vec], dim=1)
        y_pred = self.dnn(dense_input)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"y_pred": y_pred, "loss": loss}
        return return_dict


