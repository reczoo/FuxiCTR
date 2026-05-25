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


class MaskedAveragePooling(nn.Module):
    """Average pooling layer that masks out padding tokens.

    ``MaskedAveragePooling`` computes the mean of embedding vectors along a sequence dimension,
    excluding padding positions (identified by zero vectors or an explicit mask).
    """

    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix, mask=None):
        """Compute masked average pooling.

        Args:
            embedding_matrix (torch.Tensor): Embedding tensor of shape
                (batch_size, seq_len, embedding_dim).
            mask (torch.Tensor, optional): Boolean mask of shape (batch_size, seq_len).
                If None, padding is inferred from zero vectors. Default: ``None``.

        Returns:
            torch.Tensor: Pooled output of shape (batch_size, embedding_dim).
        """
        sum_out = torch.sum(embedding_matrix, dim=1)
        if mask is None:
            mask = embedding_matrix.sum(dim=-1) != 0 # zeros at padding tokens
        avg_out = sum_out / (mask.float().sum(-1, keepdim=True) + 1e-12)
        return avg_out


class MaskedSumPooling(nn.Module):
    """Sum pooling layer that sums embedding vectors along a sequence dimension.

    ``MaskedSumPooling`` computes the sum of embedding vectors, implicitly masking out
    padding tokens by assuming they are represented as zeros.
    """

    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        """Compute masked sum pooling.

        Args:
            embedding_matrix (torch.Tensor): Embedding tensor of shape
                (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Pooled output of shape (batch_size, embedding_dim).
        """
        # mask by zeros
        return torch.sum(embedding_matrix, dim=1)


class KMaxPooling(nn.Module):
    """K-max pooling layer that extracts the top-k values along a specified dimension.

    ``KMaxPooling`` selects the k largest values and returns them sorted in ascending order
    along the specified dimension.

    Args:
        k (int): Number of top values to retain.
        dim (int): Dimension along which to apply k-max pooling.

    Example::

        kmax = KMaxPooling(k=3, dim=1)
        output = kmax(X)
    """

    def __init__(self, k, dim):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, X):
        """Compute k-max pooling.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Top-k values sorted in ascending order along the specified dimension.
        """
        index = X.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        output = X.gather(self.dim, index)
        return output