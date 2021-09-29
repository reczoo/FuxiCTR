# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch import nn
import torch

class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix != 0).sum(dim=1)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return embedding_vec


class MaskedSumPooling(nn.Module):
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        # mask by zeros
        return torch.sum(embedding_matrix, dim=1)


