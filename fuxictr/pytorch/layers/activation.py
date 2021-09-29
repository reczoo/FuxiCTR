# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn

class Dice(nn.Module):
    def __init__(self, input_dim, alpha=0., eps=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + (1 - p) * self.alpha * X
        return output


