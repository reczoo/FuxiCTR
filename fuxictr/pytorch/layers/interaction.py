# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from itertools import combinations
from ...pytorch.utils import set_activation


class InnerProductLayer(nn.Module):
    # output: sum (bs x 1), bi_vector: bi-pooling (bs * dim), dot_vector (bs x f2/2), element_wise (bs x f2/2 x emb_dim)
    def __init__(self, output="sum"):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in ["sum", "bi_vector", "dot_vector", "element_wise"]:
            raise RuntimeError("InnerProductLayer output={} is not supported.".format(output))

    def forward(self, feature_emb_list):
        if self._output_type in ["sum", "bi_vector"]:
            feature_emb_tensor = torch.stack(feature_emb_list)
            sum_of_square = torch.sum(feature_emb_tensor, dim=0) ** 2  # sum then square
            square_of_sum = torch.sum(feature_emb_tensor ** 2, dim=0) # square then sum
            bi_interaction_vector = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "bi_vector":
                return bi_interaction_vector
            else:
                return torch.sum(bi_interaction_vector, dim=-1).view(-1, 1)
        elif self._output_type in ["dot_vector", "element_wise"]:
            pairs = list(combinations(feature_emb_list, 2))
            emb1 = torch.stack([p for p, _ in pairs], dim=1)
            emb2 = torch.stack([q for _, q in pairs], dim=1)
            inner_product = emb1 * emb2
            if self._output_type == "dot_vector":
                inner_product = torch.sum(inner_product, dim=2)
            return inner_product


class InnerProductLayer_v2(nn.Module):
    # output: sum (bs x 1), bi_vector: bi-pooling (bs * dim), dot_vector (bs x f2/2), element_wise (bs x f2/2 x emb_dim)
    def __init__(self, num_fields=None, output="sum"):
        super(InnerProductLayer_v2, self).__init__()
        self._output_type = output
        if output not in ["sum", "bi_vector", "dot_vector", "element_wise"]:
            raise ValueError("InnerProductLayer output={} is not supported.".format(output))
        if num_fields is None:
            if output in ["dot_vector", "element_wise"]:
                raise ValueError("num_fields is required when InnerProductLayer output={}.".format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.ByteTensor),
                                                   requires_grad=False)

    def forward(self, feature_emb):
        # TODO change sum to global_sum_pooling, chnage bi_vector to Bi_interaction_pooling, dot_vector to product_layer_pooling
        # elsement_wise to element_wise_product
        if self._output_type in ["sum", "bi_vector"]: 
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feature_emb ** 2, dim=1) # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "bi_vector":
                return bi_interaction
            else:
                return torch.sum(bi_interaction, dim=-1).view(-1, 1)
        elif self._output_type == "element_wise":
            emb1 =  torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "dot_vector":
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




