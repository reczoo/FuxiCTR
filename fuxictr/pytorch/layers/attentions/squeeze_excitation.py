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


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation layer for feature-wise recalibration.

    ``SqueezeExcitation`` computes a global statistic (mean) across the embedding
    dimension, passes it through a bottleneck MLP, and rescales each feature field.

    Args:
        num_fields (int): Number of feature fields.
        reduction_ratio (int, optional): Reduction ratio for the bottleneck layer.
            Default: ``3``.
        excitation_activation (str, optional): Activation at the output of the
            excitation network, either ``"ReLU"`` or ``"Sigmoid"``. Default: ``"ReLU"``.
    """

    def __init__(self, num_fields, reduction_ratio=3, excitation_activation="ReLU"):
        super(SqueezeExcitation, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        excitation = [nn.Linear(num_fields, reduced_size, bias=False),
                      nn.ReLU(),
                      nn.Linear(reduced_size, num_fields, bias=False)]
        if excitation_activation.lower() == "relu":
            excitation.append(nn.ReLU())
        elif excitation_activation.lower() == "sigmoid":
            excitation.append(nn.Sigmoid())
        else:
            raise NotImplementedError
        self.excitation = nn.Sequential(*excitation)

    def forward(self, feature_emb):
        """Apply squeeze-and-excitation to feature embeddings.

        Args:
            feature_emb (torch.Tensor): Feature embeddings of shape
                ``(batch_size, num_fields, embedding_dim)``.

        Returns:
            torch.Tensor: Recalibrated embeddings of the same shape as ``feature_emb``.
        """
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V
        