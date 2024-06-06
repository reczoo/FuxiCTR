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


class HolographicInteraction(nn.Module):
    def __init__(self, num_fields, interaction_type="circular_convolution"):
        super(HolographicInteraction, self).__init__()
        self.interaction_type = interaction_type
        if self.interaction_type == "circular_correlation":
            self.conj_sign =  nn.Parameter(torch.tensor([1., -1.]), requires_grad=False)
        self.triu_index = nn.Parameter(torch.triu_indices(num_fields, num_fields, offset=1), requires_grad=False)

    def forward(self, feature_emb):
        emb1 =  torch.index_select(feature_emb, 1, self.triu_index[0])
        emb2 = torch.index_select(feature_emb, 1, self.triu_index[1])
        if self.interaction_type == "hadamard_product":
            interact_tensor = emb1 * emb2
        elif self.interaction_type == "circular_convolution":
            fft1 = torch.view_as_real(torch.fft.fft(emb1))
            fft2 = torch.view_as_real(torch.fft.fft(emb2))
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(fft_product)))[..., 0]
        elif self.interaction_type == "circular_correlation":
            fft1_emb = torch.view_as_real(torch.fft.fft(emb1))
            fft1 = fft1_emb * self.conj_sign.expand_as(fft1_emb)
            fft2 = torch.view_as_real(torch.fft.fft(emb2))
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], 
                                       fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], 
                                       dim=-1)
            interact_tensor = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(fft_product)))[..., 0]
        else:
            raise ValueError("interaction_type={} not supported.".format(self.interaction_type))
        return interact_tensor
