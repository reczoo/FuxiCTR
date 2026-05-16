# =========================================================================
# Copyright (C) 2024. XiaoLongtao. All rights reserved.
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
""" This model implements the paper: Zhang et al., Wukong: Towards a Scaling Law for 
    Large-Scale Recommendation, Arxiv 2024.
    [PDF] https://arxiv.org/abs/2403.02545
"""

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class WuKong(BaseModel):
    """
    The WuKong model class that implements Meta's ICML'24 paper.

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs (e.g., vocab_size).
        model_id (str): Equivalent to model class name by default, which is used in config to determine
            which model to call. Default: ``"WuKong"``.
        gpu (int): gpu device used to load model. ``-1`` means cpu. Default: ``-1``.
        learning_rate (float): learning rate for training. Default: ``1e-3``.
        embedding_dim (int): embedding dimension of features. Default: ``64``.
        num_wukong_layers (int): number of WuKong layers. Default: ``3``.
        lcb_features (int): dimension of compressed features in LCB. Default: ``40``.
        fmb_features (int): dimension of FMB output. Default: ``40``.
        fmb_mlp_units (list): hidden MLP units of FMB. Default: ``[32, 32]``.
        fmb_mlp_activations (str): hidden MLP activations of FMB. Default: ``"relu"``.
        fmp_rank_k (int): dimension of projection matrix in FMB. Default: ``8``.
        mlp_hidden_units (list): hidden units of MLP on top of WuKong. Default: ``[32, 32]``.
        mlp_hidden_activations (str): hidden activations of MLP on top of WuKong. Default: ``"relu"``.
        mlp_batch_norm (bool): whether to apply batch normalization in MLP. Default: ``True``.
        layer_norm (bool): whether to apply layer normalization. Default: ``True``.
        net_dropout (float): dropout rate used in Wukong. Default: ``0``.
        embedding_regularizer (str or None): regularization term used for embedding parameters. Default: ``None``.
        net_regularizer (str or None): regularization term used for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="WuKong",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_wukong_layers=3,
                 lcb_features=40,
                 fmb_features=40,
                 fmb_mlp_units=[32,32],
                 fmb_mlp_activations="relu",
                 fmp_rank_k=8,
                 mlp_hidden_units=[32,32],
                 mlp_hidden_activations='relu',
                 mlp_batch_norm=True,
                 layer_norm=True,
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(WuKong, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        output_features = lcb_features + fmb_features
        self.wukong_stack = nn.Sequential(*[
            WuKongLayer(input_features=feature_map.num_fields if i == 0 else output_features,
                        lcb_features=lcb_features,
                        fmb_features=fmb_features,
                        embedding_dim=embedding_dim,
                        fmp_rank_k=fmp_rank_k,
                        fmb_mlp_units=fmb_mlp_units,
                        fmb_mlp_activations=fmb_mlp_activations,
                        fmb_dropout=net_dropout,
                        layer_norm=layer_norm) \
            for i in range(num_wukong_layers)
            ])
        self.fc = MLP_Block(input_dim=output_features * embedding_dim,
                            output_dim=1,
                            hidden_units=mlp_hidden_units,
                            hidden_activations=mlp_hidden_activations,
                            output_activation=self.output_activation,
                            batch_norm=mlp_batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of WuKong.

        Args:
            inputs: Model inputs.

        Returns:
            dict: Dictionary containing ``y_pred``.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        wukong_out = self.wukong_stack(feature_emb)
        y_pred = self.fc(wukong_out.flatten(start_dim=1))
        return_dict = {"y_pred": y_pred}
        return return_dict


class FactorizationMachineBlock(nn.Module):
    """Factorization Machine Block (FMB).

    Args:
        input_features (int): Number of input features. Default: ``16``.
        output_features (int): Number of output features. Default: ``16``.
        embedding_dim (int): Embedding dimension. Default: ``16``.
        rank_k (int): Rank for the projection matrix (None for vanilla FM). Default: ``8``.
        mlp_hidden_units (list): Hidden units of the MLP. Default: ``[16, 16]``.
        mlp_hidden_activations (str): Activation function for MLP layers. Default: ``"relu"``.
        mlp_dropout (float): Dropout rate for MLP. Default: ``0``.
    """
    def __init__(self, input_features=16, output_features=16, embedding_dim=16, rank_k=8,
                 mlp_hidden_units=[16, 16], mlp_hidden_activations="relu", mlp_dropout=0):
        super(FactorizationMachineBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_features = output_features
        self.rank_k = rank_k
        self.input_features = input_features
        if self.rank_k is not None:
            # optimized FM
            self.proj_Y = nn.Parameter(torch.randn(self.input_features, self.rank_k))
            fm_out_dim = input_features * rank_k
        else:
            # vanilla FM
            fm_out_dim = input_features * input_features
        self.layer_norm = nn.LayerNorm(fm_out_dim)
        self.mlp = MLP_Block(input_dim=fm_out_dim,
                             output_dim=output_features * embedding_dim,
                             hidden_units=mlp_hidden_units,
                             hidden_activations=mlp_hidden_activations,
                             output_activation="relu",
                             dropout_rates=mlp_dropout)

    def forward(self, x):
        """Forward pass of FMB.

        Args:
            x: Input tensor of shape ``(batch, features, embedding_dim)``.

        Returns:
            Tensor: Output of shape ``(batch, output_features, embedding_dim)``.
        """
        flatten_fm = self.optimized_fm(x)
        mlp_in = self.layer_norm(flatten_fm)
        mlp_out = self.mlp(mlp_in)
        return mlp_out.view(-1, self.output_features, self.embedding_dim)

    def optimized_fm(self, x):
        """Compute optimized or vanilla FM interaction.

        Args:
            x: Input tensor of shape ``(batch, features, embedding_dim)``.

        Returns:
            Tensor: Flattened FM matrix.
        """
        _, n, d = x.shape
        if self.rank_k is not None:
            projected = x.transpose(1, 2) @ self.proj_Y # b x d x k
            fm_matrix = torch.bmm(x, projected) # b x n x k
        else:
            fm_matrix = torch.bmm(x, x.transpose(1, 2)) # b x n x n
        return fm_matrix.flatten(start_dim=1)


class LinearCompressionBlock(nn.Module):
    """Linear Compression Block (LCB).

    Args:
        input_features (int): Number of input features. Default: ``16``.
        output_features (int): Number of output features. Default: ``8``.
    """
    def __init__(self, input_features=16, output_features=8):
        super(LinearCompressionBlock, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x):
        """Forward pass of LCB.

        Args:
            x: Input tensor of shape ``(batch, features, embedding_dim)``.

        Returns:
            Tensor: Compressed tensor of shape ``(batch, output_features, embedding_dim)``.
        """
        out = self.linear(x.transpose(1, 2))
        return out.transpose(1, 2)


class WuKongLayer(nn.Module):
    """WuKong layer combining FMB and LCB with residual connection.

    Args:
        input_features (int): Number of input features. Default: ``16``.
        lcb_features (int): Number of LCB output features. Default: ``8``.
        fmb_features (int): Number of FMB output features. Default: ``8``.
        embedding_dim (int): Embedding dimension. Default: ``16``.
        fmp_rank_k (int): Rank for FMB projection. Default: ``4``.
        fmb_mlp_units (list): Hidden units of FMB MLP. Default: ``[16, 16]``.
        fmb_mlp_activations (str): Activation function for FMB MLP. Default: ``"relu"``.
        fmb_dropout (float): Dropout rate for FMB. Default: ``0.1``.
        layer_norm (bool): Whether to apply layer normalization. Default: ``True``.
    """
    def __init__(self, input_features=16, lcb_features=8, fmb_features=8, embedding_dim=16,
                 fmp_rank_k=4, fmb_mlp_units=[16, 16], fmb_mlp_activations="relu",
                 fmb_dropout=0.1, layer_norm=True):
        super(WuKongLayer, self).__init__()
        self.fmb = FactorizationMachineBlock(input_features,
                                             fmb_features,
                                             embedding_dim,
                                             fmp_rank_k,
                                             fmb_mlp_units,
                                             fmb_mlp_activations,
                                             fmb_dropout)
        self.lcb = LinearCompressionBlock(input_features, lcb_features)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        if input_features != lcb_features + fmb_features:
            self.residual_proj = nn.Linear(input_features, lcb_features + fmb_features)

    def forward(self, x):
        """Forward pass of WuKongLayer.

        Args:
            x: Input tensor of shape ``(batch, features, embedding_dim)``.

        Returns:
            Tensor: Output tensor with residual connection.
        """
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1) # b x (fmb + lcb) x d
        out = self.residual(concat_out, x)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out

    def residual(self, out, x):
        """Apply residual connection with projection if needed.

        Args:
            out: Output tensor.
            x: Input tensor for residual.

        Returns:
            Tensor: Residual output.
        """
        if out.shape[1] != x.shape[1]:
            res = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        else:
            res = x
        return out + res
