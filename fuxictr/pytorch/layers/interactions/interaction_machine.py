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


class InteractionMachine(nn.Module):
    """Interaction Machine that computes high-order feature interactions efficiently.

    ``InteractionMachine`` uses the power-sum trick to compute interactions up to the 5th order
    without explicitly enumerating feature combinations. It supports batch normalization and
    outputs a scalar prediction via a linear layer.

    Args:
        embedding_dim (int): Dimension of the feature embeddings.
        order (int, optional): Order of interactions (must be < 6). Default: ``2``.
        batch_norm (bool, optional): Whether to apply batch normalization. Default: ``False``.

    Example::

        im = InteractionMachine(embedding_dim=16, order=3, batch_norm=True)
        output = im(feature_emb)
    """

    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, "order={} is not supported.".format(order)
        self.order = order
        self.bn = nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = nn.Linear(order * embedding_dim, 1)

    def second_order(self, p1, p2):
        """Compute second-order interactions from power sums.

        Args:
            p1 (torch.Tensor): First power sum.
            p2 (torch.Tensor): Second power sum.

        Returns:
            torch.Tensor: Second-order interaction terms.
        """
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        """Compute third-order interactions from power sums.

        Args:
            p1 (torch.Tensor): First power sum.
            p2 (torch.Tensor): Second power sum.
            p3 (torch.Tensor): Third power sum.

        Returns:
            torch.Tensor: Third-order interaction terms.
        """
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        """Compute fourth-order interactions from power sums.

        Args:
            p1 (torch.Tensor): First power sum.
            p2 (torch.Tensor): Second power sum.
            p3 (torch.Tensor): Third power sum.
            p4 (torch.Tensor): Fourth power sum.

        Returns:
            torch.Tensor: Fourth-order interaction terms.
        """
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2)
                + 8 * p1 * p3 - 6 * p4) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        """Compute fifth-order interactions from power sums.

        Args:
            p1 (torch.Tensor): First power sum.
            p2 (torch.Tensor): Second power sum.
            p3 (torch.Tensor): Third power sum.
            p4 (torch.Tensor): Fourth power sum.
            p5 (torch.Tensor): Fifth power sum.

        Returns:
            torch.Tensor: Fifth-order interaction terms.
        """
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4
                - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120

    def forward(self, X):
        """Compute high-order feature interactions.

        Args:
            X (torch.Tensor): Feature embeddings of shape
                (batch_size, num_fields, embedding_dim).

        Returns:
            torch.Tensor: Scalar prediction of shape (batch_size, 1).
        """
        out = []
        Q = X
        if self.order >= 1:
            p1 = Q.sum(dim=1)
            out.append(p1)
            if self.order >= 2:
                Q = Q * X
                p2 = Q.sum(dim=1)
                out.append(self.second_order(p1, p2))
                if self.order >= 3:
                    Q = Q * X
                    p3 = Q.sum(dim=1)
                    out.append(self.third_order(p1, p2, p3))
                    if self.order >= 4:
                        Q = Q * X
                        p4 = Q.sum(dim=1)
                        out.append(self.fourth_order(p1, p2, p3, p4))
                        if self.order == 5:
                            Q = Q * X
                            p5 = Q.sum(dim=1)
                            out.append(self.fifth_order(p1, p2, p3, p4, p5))
        out = torch.cat(out, dim=-1)
        if self.bn is not None:
            out = self.bn(out)
        y = self.fc(out)
        return y