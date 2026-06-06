# =========================================================================
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


import sys
sys.path.append('../../')
import torch
from fuxictr.pytorch.layers import BilinearInteraction, BilinearInteractionV2

num_fields = 3
embedding_dim = 4

bilinear_types = ["field_all", "field_each", "field_interaction"]
bilinear_W_list = [torch.rand(embedding_dim, embedding_dim),
                   torch.rand(num_fields, embedding_dim, embedding_dim),
                   torch.rand(int(num_fields * (num_fields - 1) / 2), embedding_dim, embedding_dim)]
feature_emb = torch.rand(2, num_fields, embedding_dim)

for bilinear_type, bilinear_W in zip(bilinear_types, bilinear_W_list):
    bilinear_interaction = BilinearInteraction(num_fields, embedding_dim, bilinear_type)
    bilinear_interaction.bilinear_W.data = bilinear_W
    bilinear_out = bilinear_interaction(feature_emb)
    # print(bilinear_out)

    bilinear_interaction2 = BilinearInteractionV2(num_fields, embedding_dim, bilinear_type)
    bilinear_interaction2.bilinear_W.data = bilinear_W
    bilinear_out2 = bilinear_interaction2(feature_emb)
    # print(bilinear_out2)
    
    assert torch.allclose(bilinear_out, bilinear_out2)


