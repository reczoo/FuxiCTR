# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CompressedInteractionNet, LogisticRegression


class xDeepFM(BaseModel):
    """xDeepFM model with Compressed Interaction Network (CIN) and DNN.

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs.
        model_id (str): Model identifier string. Default: ``"xDeepFM"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for training. Default: ``1e-3``.
        embedding_dim (int): Embedding dimension of features. Default: ``10``.
        dnn_hidden_units (list): Hidden units of the DNN. Default: ``[64, 64, 64]``.
        dnn_activations (str): Activation function for DNN layers. Default: ``"ReLU"``.
        cin_hidden_units (list): Hidden units of the CIN. Default: ``[16, 16, 16]``.
        net_dropout (float): Dropout rate for DNN. Default: ``0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network weights. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="xDeepFM",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 cin_hidden_units=[16, 16, 16],
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(xDeepFM, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only CIN used
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        self.cin = CompressedInteractionNet(feature_map.num_fields, cin_hidden_units, output_dim=1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of xDeepFM.

        Args:
            inputs: Model inputs.

        Returns:
            dict: Dictionary containing ``y_pred``.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # list of b x embedding_dim
        lr_logit = self.lr_layer(X)
        cin_logit = self.cin(feature_emb)
        y_pred = lr_logit + cin_logit # only LR + CIN
        if self.dnn is not None:
            dnn_logit = self.dnn(feature_emb.flatten(start_dim=1))
            y_pred += dnn_logit # LR + CIN + DNN
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


