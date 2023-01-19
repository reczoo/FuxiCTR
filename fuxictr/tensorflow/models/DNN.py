# =========================================================================
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
from fuxictr.tensorflow.layers import FeatureEmbedding, MLP_Block


class DNN(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="DNN", 
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DNN, self).__init__(feature_map, model_id=model_id, **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim,
                                                embedding_regularizer=embedding_regularizer)
        self.emb_out_dim = feature_map.sum_emb_out_dim()
        self.mlp = MLP_Block(input_dim=self.emb_out_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm,
                             regularizer=net_regularizer)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
    
    def call(self, inputs, training=False):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.mlp(tf.reshape(feature_emb, (-1, self.emb_out_dim)))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
        
