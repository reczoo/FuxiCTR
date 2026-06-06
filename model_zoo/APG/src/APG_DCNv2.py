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

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, FeatureEmbeddingDict, CrossNetV2, CrossNetMix
from .APG import APG_MLP


class APG_DCNv2(BaseModel):
    """Adaptive Parameter Generation DCNv2 model.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"APG_DCNv2"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        model_structure (str): Model structure, one of ["crossnet_only", "stacked", "parallel", "stacked_parallel"]. Default: ``"parallel"``.
        use_low_rank_mixture (bool): Whether to use low-rank mixture cross network. Default: ``False``.
        low_rank (int): Low rank for mixture cross network. Default: ``32``.
        num_experts (int): Number of experts for mixture cross network. Default: ``4``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        stacked_dnn_hidden_units (list): Hidden units for stacked DNN. Default: ``[]``.
        parallel_dnn_hidden_units (list): Hidden units for parallel DNN. Default: ``[]``.
        dnn_activations (str): Activation functions for DNN. Default: ``"ReLU"``.
        num_cross_layers (int): Number of cross layers. Default: ``3``.
        net_dropout (float): Dropout rate for network. Default: ``0``.
        batch_norm (bool): Whether to use batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        hypernet_config (dict): Configuration dict for hypernetwork. Default: ``{}``.
        condition_features (list): List of condition feature names. Default: ``[]``.
        condition_mode (str): Conditioning mode, one of ["self-wise", "group-wise", "mix-wise"]. Default: ``"self-wise"``.
        new_condition_emb (bool): Whether to use a separate embedding layer for condition features. Default: ``False``.
        rank_k (int): Rank for low-rank weight generation. Default: ``32``.
        overparam_p (int): Over-parameterization dimension. Default: ``1024``.
        generate_bias (bool): Whether to generate bias via hypernetwork. Default: ``True``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="APG_DCNv2",
                 gpu=-1,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 stacked_dnn_hidden_units=[],
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hypernet_config={},
                 condition_features=[],
                 condition_mode="self-wise",
                 new_condition_emb=False,
                 rank_k=32,
                 overparam_p=1024,
                 generate_bias=True,
                 **kwargs):
        super(APG_DCNv2, self).__init__(feature_map,
                                        model_id=model_id,
                                        gpu=gpu,
                                        embedding_regularizer=embedding_regularizer,
                                        net_regularizer=net_regularizer,
                                        **kwargs)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.condition_mode = condition_mode
        self.condition_features = condition_features
        self.condition_emb_layer = None
        if condition_mode == "self-wise":
            condition_dim = None
        else:
            assert len(condition_features) > 0
            condition_dim = len(condition_features) * embedding_dim
            if new_condition_emb:
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim,
                    required_feature_columns=condition_features)
        input_dim = feature_map.sum_emb_out_dim()
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(input_dim, num_cross_layers, low_rank=low_rank,
                                        num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel",
            "stacked_parallel"], "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = APG_MLP(input_dim=input_dim,
                                       output_dim=None, 
                                       hidden_units=stacked_dnn_hidden_units,
                                       hidden_activations=dnn_activations,
                                       output_activation=None, 
                                       dropout_rates=net_dropout, 
                                       batch_norm=batch_norm,
                                       hypernet_config=hypernet_config,
                                       condition_dim=condition_dim,
                                       condition_mode=condition_mode,
                                       rank_k=rank_k,
                                       overparam_p=overparam_p,
                                       generate_bias=generate_bias)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = APG_MLP(input_dim=input_dim,
                                        output_dim=None, 
                                        hidden_units=parallel_dnn_hidden_units,
                                        hidden_activations=dnn_activations,
                                        output_activation=None, 
                                        dropout_rates=net_dropout, 
                                        batch_norm=batch_norm,
                                        hypernet_config=hypernet_config,
                                        condition_dim=condition_dim,
                                        condition_mode=condition_mode,
                                        rank_k=rank_k,
                                        overparam_p=overparam_p,
                                        generate_bias=generate_bias)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
        self.fc = nn.Linear(final_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of APG_DCNv2.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        condition_z = self.get_condition_z(X, feature_emb_dict)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        cross_out = self.crossnet(feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out, condition_z)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb, condition_z)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out, condition_z),
                                   self.parallel_dnn(feature_emb, condition_z)], dim=-1)
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_condition_z(self, X, feature_emb_dict):
        """Get condition vector from input features.

        Args:
            X: Input feature dict.
            feature_emb_dict: Feature embedding dict.

        Returns:
            torch.Tensor or None: Condition vector or None if self-wise mode.
        """
        condition_z = None
        if self.condition_mode != "self-wise":
            if self.condition_emb_layer is not None:
                condition_z = self.condition_emb_layer(X, flatten_emb=True)
            else:
                condition_z = self.embedding_layer.dict2tensor(feature_emb_dict,
                                                               feature_list=self.condition_features,
                                                               flatten_emb=True)
        return condition_z