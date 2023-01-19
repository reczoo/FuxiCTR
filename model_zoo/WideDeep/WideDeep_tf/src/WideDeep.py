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
from fuxictr.tensorflow.layers import FeatureEmbedding, MLP_Block, LogisticRegression
from fuxictr.tensorflow.tf_utils import get_loss, get_optimizer
from tensorflow.keras import optimizers


class WideDeep(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="WideDeep", 
                 wide_learning_rate=1e-3, 
                 deep_learning_rate=1e-3, 
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(WideDeep, self).__init__(feature_map, model_id=model_id, **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim,
                                                embedding_regularizer=embedding_regularizer)
        self.lr_layer = LogisticRegression(feature_map, use_bias=True, 
                                           regularizer=embedding_regularizer)
        self.emb_out_dim = feature_map.sum_emb_out_dim()
        self.mlp = MLP_Block(input_dim=self.emb_out_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm,
                             regularizer=net_regularizer)
        self.compile(kwargs["loss"], wide_learning_rate, deep_learning_rate, kwargs["deep_optimizer"])

    def compile(self, loss="bce", wide_lr=1.e-4, deep_lr=1.e-3, deep_optimizer="adam"):
        super(BaseModel, self).compile(optimizer=[optimizers.Ftrl(learning_rate=wide_lr, l1_regularization_strength=0.1),
                                                  get_optimizer(deep_optimizer, deep_lr)],
                                       loss=get_loss(loss))

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        self.optimizer[1].learning_rate = max(self.optimizer[1].learning_rate * factor, min_lr)
        return self.optimizer[1].lr.numpy()

    @tf.function
    def train_step(self, batch_data):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.get_total_loss(batch_data)
            wide_prefix = "logistic"
            wide_variables = [var for var in self.trainable_variables if wide_prefix in var.name]
            wide_grads = tape.gradient(loss, wide_variables)
            wide_grads, _ = tf.clip_by_global_norm(wide_grads, self._max_gradient_norm)
            self.optimizer[0].apply_gradients(zip(wide_grads, wide_variables))
            deep_variables = [var for var in self.trainable_variables if wide_prefix not in var.name]
            deep_grads = tape.gradient(loss, deep_variables)
            deep_grads, _ = tf.clip_by_global_norm(deep_grads, self._max_gradient_norm)
            self.optimizer[1].apply_gradients(zip(deep_grads, deep_variables))
        return loss

    def call(self, inputs, training=False):
        X = self.get_inputs(inputs)
        y_pred = self.lr_layer(X)
        feature_emb = self.embedding_layer(X)
        y_pred += self.mlp(tf.reshape(feature_emb, [-1, self.emb_out_dim]))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

