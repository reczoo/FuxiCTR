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


from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class ONNv2(BaseModel):
    """Operation-aware Neural Network v2 (ONNv2), also known as NFFM/DeepFFM (PyTorch version).

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs.
        model_id (str): Model identifier string. Default: ``"ONNv2"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for training. Default: ``1e-3``.
        embedding_dim (int): Embedding dimension of features. Default: ``2``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network weights. Default: ``None``.
        hidden_units (list): Hidden units of the DNN. Default: ``[64, 64, 64]``.
        hidden_activations (str): Activation function for DNN layers. Default: ``"ReLU"``.
        net_dropout (float): Dropout rate for DNN. Default: ``0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``False``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="ONNv2",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=2,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 **kwargs):
        super(ONNv2, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.num_fields = feature_map.num_fields
        self.embedding_dim = embedding_dim
        self.interact_units = int(self.num_fields * (self.num_fields - 1) / 2)
        self.dnn = MLP_Block(input_dim=embedding_dim * self.num_fields + self.interact_units,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim * self.num_fields) # b x f x dim*f
        self.diag_mask = torch.eye(self.num_fields).bool().to(self.device)
        self.triu_mask = torch.triu(torch.ones(self.num_fields, self.num_fields), 1).bool().to(self.device)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of ONNv2.

        Args:
            inputs: Model inputs.

        Returns:
            dict: Dictionary containing ``y_pred``.
        """
        X = self.get_inputs(inputs)
        field_wise_emb = self.embedding_layer(X).view(-1, self.num_fields, self.num_fields, self.embedding_dim)
        batch_size = field_wise_emb.shape[0]
        diag_embedding = torch.masked_select(field_wise_emb, self.diag_mask.unsqueeze(-1)).view(batch_size, -1)
        ffm_out = self.ffm_interaction(field_wise_emb)
        dnn_input = torch.cat([diag_embedding, ffm_out], dim=1)
        y_pred = self.dnn(dnn_input)
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
        out = (field_wise_emb.transpose(1, 2) * field_wise_emb).sum(dim=-1)
        out = torch.masked_select(out, self.triu_mask).view(-1, self.interact_units)
        return out
