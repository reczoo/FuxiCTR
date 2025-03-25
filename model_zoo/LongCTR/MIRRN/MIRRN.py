# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2024. psycho-demon@Github.
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
import torch.nn.functional as F
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import (
    FeatureEmbedding,
    MLP_Block,
    MultiHeadTargetAttention
)
from fuxictr.utils import not_in_whitelist


class MIRRN(BaseModel):
    """
    Ref: https://github.com/USTC-StarTeam/MIRRN
    """
    def __init__(self,
                 feature_map,
                 model_id="MIRRN",
                 gpu=-1,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dim=64,
                 num_heads=1,
                 use_scale=True,
                 attention_dropout=0,
                 reuse_hash=True,
                 hash_bits=32,
                 topk=50,
                 max_len=1000,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 short_seq_len=50,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(MIRRN, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.reuse_hash = reuse_hash
        self.hash_bits = hash_bits
        self.topk = topk
        self.short_seq_len = short_seq_len
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.short_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout,
                                                        use_scale)
        self.pos = nn.Embedding(max_len + 1, self.item_info_dim)
        self.random_rotations = nn.Parameter(torch.randn(self.item_info_dim, self.hash_bits),
                                             requires_grad=False)
        self.MHFT_block = nn.ModuleList()
        self.MHFT_block.append(FilterLayer2(topk, self.item_info_dim, 0.1, 4))
        self.MHFT_block.append(FilterLayer2(topk, self.item_info_dim, 0.1, 4))
        self.MHFT_block.append(FilterLayer2(topk, self.item_info_dim, 0.1, 4))
        self.long_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                       attention_dim,
                                                       num_heads,
                                                       attention_dropout,
                                                       use_scale)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + self.item_info_dim * 2,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            emb_out = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(emb_out)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        # short interest attention
        target_emb = item_feat_emb[:, -1, :]
        short_seq_emb = item_feat_emb[:, -self.short_seq_len:-1, :]
        short_mask = mask[:, -self.short_seq_len:-1]
        short_interest = self.short_attention(target_emb, short_seq_emb, short_mask)

        # long interest attention
        sequence_emb = item_feat_emb[:, 0:-1, :]
        # target
        topk_target_emb, topk_target_mask, topk_target_index = self.topk_retrieval(
            self.random_rotations,
            target_emb, sequence_emb, mask,
            self.topk
        )
        # short
        short_emb = sequence_emb[:, -16:]
        mean_short_emb = self.masked_mean(short_emb, mask[:, -16:], dim=1)
        topk_short_emb, topk_short_mask, topk_short_index = self.topk_retrieval(
            self.random_rotations,
            mean_short_emb, sequence_emb, mask,
            self.topk
        )
        # global
        mean_global_emb = self.masked_mean(sequence_emb, mask, dim=1)
        topk_global_emb, topk_global_mask, topk_global_index = self.topk_retrieval(
            self.random_rotations,
            mean_global_emb, sequence_emb,
            mask, self.topk
        )
        # pos
        pos_mask_target = sequence_emb.shape[1] - topk_target_index
        pos_target = self.pos(pos_mask_target)
        topk_target_emb += pos_target * 0.02

        pos_mask_short = sequence_emb.shape[1] - topk_short_index
        pos_short = self.pos(pos_mask_short)
        topk_short_emb += pos_short * 0.02

        pos_mask_global = sequence_emb.shape[1] - topk_global_index
        pos_global = self.pos(pos_mask_global)
        topk_global_emb += pos_global * 0.02
        # MHFT
        target_interest_emb = self.MHFT_block[0](topk_target_emb).mean(1)
        short_interest_emb = self.MHFT_block[1](topk_short_emb).mean(1)
        global_interest_emb = self.MHFT_block[2](topk_global_emb).mean(1)

        interest_emb = torch.stack((target_interest_emb, short_interest_emb, global_interest_emb), 1)
        long_interest = self.long_attention(target_emb, interest_emb)
        emb_list += [target_emb, short_interest, long_interest]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict
    
    def masked_mean(self, tensor, mask, dim=1):
        mask = mask.unsqueeze(-1)
        masked_sum = (tensor * mask).sum(dim)
        masked_count = mask.sum(dim)
        return masked_sum / (masked_count + 1e-9)

    def topk_retrieval(self, random_rotations, target_item, history_sequence, mask, topk=5):
        if not self.reuse_hash:
            random_rotations = torch.randn(target_item.size(1), self.hash_bits, device=target_item.device)
        target_hash = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
        sequence_hash = self.lsh_hash(history_sequence, random_rotations)
        hash_sim = -torch.abs(sequence_hash - target_hash).sum(dim=-1)
        hash_sim = hash_sim.masked_fill_(mask.float() == 0, -(self.hash_bits + 1))
        topk = min(topk, hash_sim.shape[1]) # make sure input seq_len >= topk
        topk_index = hash_sim.topk(topk, dim=1, largest=True, sorted=True)[1]
        topk_index = topk_index.sort(-1)[0]
        topk_emb = torch.gather(history_sequence, 1,
                                topk_index.unsqueeze(-1).expand(-1, -1, history_sequence.shape[-1]))
        topk_mask = torch.gather(mask, 1, topk_index)
        return topk_emb, topk_mask, topk_index

    def lsh_hash(self, vecs, random_rotations):
        """ See the tensorflow-lsh-functions for reference:
            https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py

            Input: vecs, with hape B x seq_len x d
        """
        rotated_vecs = torch.matmul(vecs, random_rotations)  # B x seq_len x num_hashes
        hash_code = torch.relu(torch.sign(rotated_vecs))
        return hash_code

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss


class FilterLayer2(nn.Module):
    def __init__(self, max_length, hidden_size, hidden_dropout_prob, n_block):
        super(FilterLayer2, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(n_block, hidden_size // n_block, 
                        hidden_size // n_block, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.n = n_block

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        A = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        A = A.view(batch, seq_len // 2 + 1, self.n, hidden // self.n)
        B = torch.view_as_complex(self.complex_weight)
        C = torch.einsum("blnd,ndd->blnd", A, B)
        C = C.view(batch, seq_len // 2 + 1, hidden)
        sequence_emb_fft = torch.fft.irfft(C, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """ Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
