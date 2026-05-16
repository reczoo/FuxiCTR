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
import numpy as np


class Dice(nn.Module):
    """Data-adaptive activation function used in DIN and related models.

    ``Dice`` applies batch normalization followed by a sigmoid gating mechanism
    with a learnable parameter ``alpha``.

    Args:
        input_dim (int): Dimensionality of the input features.
        eps (float, optional): Epsilon for batch normalization. Default: ``1e-9``.
    """

    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        """Apply the Dice activation to the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            torch.Tensor: Activated tensor of the same shape as ``X``.
        """
        p = torch.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) *  X
        return output


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function.

    Reference: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        """Apply the GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor of the same shape as ``x``.
        """
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * ( x + 0.044715 * torch.pow(x, 3))))
