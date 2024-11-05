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
from itertools import combinations


class BilinearInteraction(nn.Module):
    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.interact_dim = int(num_fields * (num_fields - 1) / 2)
        if self.bilinear_type == "field_all":
            self.bilinear_W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        elif self.bilinear_type == "field_each":
            self.bilinear_W = nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        elif self.bilinear_type == "field_interaction":
            self.bilinear_W = nn.Parameter(torch.Tensor(self.interact_dim, embedding_dim, embedding_dim))
        else:
            raise NotImplementedError
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.bilinear_W)

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [torch.matmul(v_i, self.bilinear_W) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_W_list = torch.split(self.bilinear_W, 1, dim=0)
            bilinear_list = [torch.matmul(feature_emb_list[i], bilinear_W_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_W_list = torch.split(self.bilinear_W, 1, dim=0)
            bilinear_list = [torch.matmul(v[0], bilinear_W_list[i]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class BilinearInteractionV2(nn.Module):
    def __init__(self, num_fields, embedding_dim, bilinear_type="field_interaction"):
        super(BilinearInteractionV2, self).__init__()
        self.bilinear_type = bilinear_type
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.interact_dim = int(num_fields * (num_fields - 1) / 2)
        if self.bilinear_type == "field_all":
            self.bilinear_W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        elif self.bilinear_type == "field_each":
            self.bilinear_W = nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        elif self.bilinear_type == "field_interaction":
            self.bilinear_W = nn.Parameter(torch.Tensor(self.interact_dim, embedding_dim, embedding_dim))
        else:
            raise NotImplementedError
        self.triu_index = nn.Parameter(torch.triu_indices(num_fields, num_fields, offset=1), requires_grad=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.bilinear_W)

    def forward(self, feature_emb):
        if self.bilinear_type == "field_interaction":
            left_emb =  torch.index_select(feature_emb, 1, self.triu_index[0])
            right_emb = torch.index_select(feature_emb, 1, self.triu_index[1])
            bilinear_out = torch.matmul(left_emb.unsqueeze(2), self.bilinear_W).squeeze(2) * right_emb
        else:
            if self.bilinear_type == "field_all":
                hidden_emb = torch.matmul(feature_emb, self.bilinear_W)
            elif self.bilinear_type == "field_each":
                hidden_emb = torch.matmul(feature_emb.unsqueeze(2), self.bilinear_W).squeeze(2)
            left_emb =  torch.index_select(hidden_emb, 1, self.triu_index[0])
            right_emb = torch.index_select(feature_emb, 1, self.triu_index[1])
            bilinear_out = left_emb * right_emb
        return bilinear_out

