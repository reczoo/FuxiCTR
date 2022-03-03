# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from itertools import combinations
from ..torch_utils import get_activation


class InnerProductLayer(nn.Module):
    """ output: product_sum_pooling (bs x 1), 
                Bi_interaction_pooling (bs * dim), 
                inner_product (bs x f2/2), 
                elementwise_product (bs x f2/2 x emb_dim)
    """
    def __init__(self, num_fields=None, output="product_sum_pooling"):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in ["product_sum_pooling", "Bi_interaction_pooling", "inner_product", "elementwise_product"]:
            raise ValueError("InnerProductLayer output={} is not supported.".format(output))
        if num_fields is None:
            if output in ["inner_product", "elementwise_product"]:
                raise ValueError("num_fields is required when InnerProductLayer output={}.".format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.ByteTensor),
                                                   requires_grad=False)

    def forward(self, feature_emb):
        if self._output_type in ["product_sum_pooling", "Bi_interaction_pooling"]: 
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feature_emb ** 2, dim=1) # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "Bi_interaction_pooling":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "elementwise_product":
            emb1 =  torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "inner_product":
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)


class BilinearInteractionLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                                 for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                                 for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class HolographicInteractionLayer(nn.Module):
    def __init__(self, num_fields, interaction_type="circular_convolution"):
        super(HolographicInteractionLayer, self).__init__()
        self.interaction_type = interaction_type
        if self.interaction_type == "circular_correlation":
            self.conj_sign =  nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
        self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)

    def forward(self, feature_emb):
        emb1 =  torch.index_select(feature_emb, 1, self.field_p)
        emb2 = torch.index_select(feature_emb, 1, self.field_q)
        if self.interaction_type == "hadamard_product":
            interact_tensor = emb1 * emb2
        elif self.interaction_type == "circular_convolution":
            fft1 = torch.rfft(emb1, 1, onesided=False)
            fft2 = torch.rfft(emb2, 1, onesided=False)
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.irfft(fft_product, 1, onesided=False)
        elif self.interaction_type == "circular_correlation":
            fft1_emb = torch.rfft(emb1, 1, onesided=False) 
            fft1 = fft1_emb * self.conj_sign.expand_as(fft1_emb)
            fft2 = torch.rfft(emb2, 1, onesided=False)
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.irfft(fft_product, 1, onesided=False)
        else:
            raise ValueError("interaction_type={} not supported.".format(self.interaction_type))
        return interact_tensor


class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i
        

class CompressedInteractionNet(nn.Module):
    def __init__(self, num_fields, cin_layer_units, output_dim=1):
        super(CompressedInteractionNet, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, feature_emb):
        pooling_outputs = []
        X_0 = feature_emb
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        output = self.fc(concate_vec)
        return output


class InteractionMachine(nn.Module):
    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, "order={} is not supported.".format(order)
        self.order = order
        self.bn = nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = nn.Linear(order * embedding_dim, 1)
        
    def second_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2)
                + 8 * p1 * p3 - 6 * p4) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4
                - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120

    def forward(self, X):
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