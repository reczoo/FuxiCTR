# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import tensorflow as tf
from fuxictr.tensorflow.models import BaseModel
from fuxictr.tensorflow.layers import FeatureEmbedding, MLP_Block, CrossNet, Linear


class DCN(BaseModel):
    """Deep & Cross Network (DCN) model (TensorFlow implementation).

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"DCN"``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        dnn_hidden_units (list): Hidden units for the DNN tower. Default: ``[]``.
        dnn_activations (str): Activation functions for DNN. Default: ``"ReLU"``.
        net_dropout (float): Dropout rate for the network. Default: ``0``.
        num_cross_layers (int): Number of cross layers. Default: ``3``.
        batch_norm (bool): Whether to use batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="DCN",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 net_dropout=0,
                 num_cross_layers=3,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DCN, self).__init__(feature_map, model_id=model_id, **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim,
                                                embedding_regularizer=embedding_regularizer)
        input_dim = feature_map.sum_emb_out_dim()
        self.crossnet = CrossNet(input_dim, num_cross_layers)
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             regularizer=net_regularizer) \
                   if dnn_hidden_units else None # in case of only crossing net used
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = Linear(1, regularizer=net_regularizer) # [cross_part, dnn_part] -> logit
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)

    def call(self, inputs, training=False):
        """Forward pass of DCN.

        Args:
            inputs: Input data containing features.
            training: Whether in training mode.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = tf.concat([cross_out, dnn_out], axis=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
