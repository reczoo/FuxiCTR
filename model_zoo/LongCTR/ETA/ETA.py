# =========================================================================
# Copyright (C) 2025. The FuxiCTR Library. All rights reserved.
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
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention
from fuxictr.utils import not_in_whitelist


class ETA(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ETA", 
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
                 learning_rate=1e-3,
                 embedding_dim=10, 
                 net_dropout=0,
                 batch_norm=False,
                 short_seq_len=50,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(ETA, self).__init__(feature_map,
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
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.short_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout,
                                                        use_scale)
        self.random_rotations = nn.Parameter(
            torch.randn(1, self.item_info_dim, self.hash_bits), 
            requires_grad=False
        )
        self.long_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                       attention_dim,
                                                       num_heads,
                                                       attention_dropout,
                                                       use_scale)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim * 2
        self.dnn = MLP_Block(input_dim=input_dim,
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
        short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)
        # long interest attention
        long_seq_emb = item_feat_emb[:, 0:-1, :]
        topk_emb, topk_mask = self.topk_retrieval(self.random_rotations, 
                                                  target_emb, long_seq_emb, mask, self.topk)
        long_interest_emb = self.long_attention(target_emb, topk_emb, topk_mask)
        emb_list += [target_emb, short_interest_emb, long_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def topk_retrieval(self, random_rotations, target_item, history_sequence, mask, topk=5):
        if self.reuse_hash:
            random_rotations = random_rotations.repeat(target_item.size(0), 1, 1)
        else:
            random_rotations = torch.randn(
                target_item.size(0), target_item.size(1), self.hash_bits,
                device=target_item.device
            )
        sequence_hash = self.lsh_hash(history_sequence, random_rotations)
        target_hash = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
        hash_dis = torch.abs(sequence_hash - target_hash).sum(dim=-1)
        hash_dis = hash_dis.masked_fill_(mask.float() == 0, 1 + self.hash_bits)
        topk = min(topk, hash_dis.shape[1]) # make sure input seq_len >= topk
        topk_index = hash_dis.topk(topk, dim=1, largest=False, sorted=True)[1]
        topk_emb = torch.gather(history_sequence, 1, 
                                topk_index.unsqueeze(-1).expand(-1, -1, history_sequence.shape[-1]))
        topk_mask = torch.gather(mask, 1, topk_index)
        return topk_emb, topk_mask
        
    def lsh_hash(self, vecs, random_rotations):
        """ See the tensorflow-lsh-functions for reference:
            https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py
            
            Input: vecs, with shape B x seq_len x d
            Output: hash_code, with shape B x seq_len x hash_bits
        """
        rotated_vecs = torch.einsum("bld,bdh->blh", vecs, random_rotations).unsqueeze(-1)
        rotated_vecs = torch.cat([-rotated_vecs, rotated_vecs], dim=-1)
        hash_code = torch.argmax(rotated_vecs, dim=-1).float()
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
