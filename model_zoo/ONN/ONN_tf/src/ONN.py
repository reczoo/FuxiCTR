# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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


class ONN(BaseModel):
    """Operation-aware Neural Network (ONN), also known as NFFM/DeepFFM.

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs.
        model_id (str): Model identifier string. Default: ``"ONN"``.
        learning_rate (float): Learning rate for training. Default: ``1e-3``.
        embedding_dim (int): Embedding dimension of features. Default: ``10``.
        hidden_units (list): Hidden units of the DNN. Default: ``[64, 64, 64]``.
        hidden_activations (str): Activation function for DNN layers. Default: ``"ReLU"``.
        net_dropout (float): Dropout rate for DNN. Default: ``0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network weights. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="ONN",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(ONN, self).__init__(feature_map, model_id=model_id, **kwargs)
        self.num_fields = feature_map.num_fields
        self.embedding_dim = embedding_dim
        self.interact_units = int(self.num_fields * (self.num_fields - 1) / 2)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim * self.num_fields,
                                                embedding_regularizer=embedding_regularizer) # b x f x dim*f
        self.emb_out_dim = embedding_dim * self.num_fields + self.interact_units,
        self.mlp = MLP_Block(input_dim=self.emb_out_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             regularizer=net_regularizer)
        self.diag_mask = tf.eye(self.num_fields, dtype=tf.bool)
        self.triu_mask = tf.linalg.band_part(tf.ones(shape=(self.num_fields, self.num_fields)), 0, -1) - tf.eye(self.num_fields)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)

    def call(self, inputs, training=False):
        """Forward pass of ONN.

        Args:
            inputs: Model inputs.
            training: Whether in training mode.

        Returns:
            dict: Dictionary containing ``y_pred``.
        """
        X = self.get_inputs(inputs)
        field_wise_emb = tf.reshape(self.embedding_layer(X), (-1, self.num_fields, self.num_fields, self.embedding_dim))
        diag_embedding = tf.boolean_mask(tf.transpose(field_wise_emb, (1, 2, 3, 0)), self.diag_mask)
        diag_embedding = tf.reshape(tf.transpose(diag_embedding, (2, 0, 1)), [-1, self.num_fields * self.embedding_dim])
        ffm_out = self.ffm_interaction(field_wise_emb)
        dnn_input = tf.concat([diag_embedding, ffm_out], axis=-1)
        y_pred = self.mlp(dnn_input, training=training)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def ffm_interaction(self, field_wise_emb):
        """Compute FFM interaction terms.

        Args:
            field_wise_emb: Field-wise embeddings of shape ``(batch, fields, fields, dim)``.

        Returns:
            Tensor: FFM interaction output.
        """
        out = tf.reduce_sum(field_wise_emb * tf.transpose(field_wise_emb, (0, 2, 1, 3)), axis=-1) # b x f x f
        out = tf.boolean_mask(tf.transpose(out, (1, 2, 0)), self.triu_mask)
        out = tf.reshape(tf.transpose(out, (1, 0)), [-1, self.interact_units]) # b x (f*(f-1)/2)
        return out

    def ffm_bi_interaction(self, field_wise_emb):
        """Compute FFM bi-interaction (alternative when interact_units is too large).

        Args:
            field_wise_emb: Field-wise embeddings.

        Returns:
            Tensor: FFM bi-interaction output.
        """
        # when interact_units is too large, consider to use ffm_bi_interaction
        out = field_wise_emb * tf.transpose(field_wise_emb, (0, 2, 1, 3)) # b x f x f x d
        out = tf.boolean_mask(tf.transpose(out, (1, 2, 3, 0)), self.triu_mask)
        out = tf.reduce_sum(tf.transpose(out, (2, 0, 1)), axis=1) # b x d
        return out

        