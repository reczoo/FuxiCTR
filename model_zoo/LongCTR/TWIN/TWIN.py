# =========================================================================
# Copyright (C) 2025. The FuxiCTR Library. All rights reserved.
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
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention
from fuxictr.utils import not_in_whitelist


class TWIN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="TWIN", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dropout=0,
                 attention_dim=64,
                 num_heads=1,
                 short_seq_len=50,
                 topk=50,
                 Kc_cross_features=0,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(TWIN, self).__init__(feature_map,
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
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
                                                        attention_dropout)
        self.long_attention = MultiHeadTopKAttention(self.item_info_dim,
                                                     Kc_cross_features,
                                                     embedding_dim,
                                                     attention_dim,
                                                     topk,
                                                     num_heads,
                                                     attention_dropout)
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
        target_emb = item_feat_emb[:, -1, :]

        # short interest attention
        short_seq_emb = item_feat_emb[:, -self.short_seq_len:-1, :]
        short_mask = mask[:, -self.short_seq_len:-1]
        short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)

        # long interest attention
        long_seq_emb = item_feat_emb[:, 0:-1, :]
        long_interest_emb = self.long_attention(target_emb, long_seq_emb, mask)
        emb_list += [target_emb, short_interest_emb, long_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

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


class MultiHeadTopKAttention(nn.Module):
    def __init__(self,
                 input_dim=64,
                 Kc=0,
                 embedding_dim=16,
                 attention_dim=64,
                 topk=50,
                 num_heads=1,
                 dropout_rate=0):
        super(MultiHeadTopKAttention, self).__init__()
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.topk = topk
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.Kc = Kc
        self.Kc_dim = Kc * embedding_dim
        self.Kh_dim = input_dim - self.Kc_dim # split item features and cross features
        self.W_q = nn.Linear(self.Kh_dim, attention_dim, bias=False)
        self.W_h = nn.Linear(self.Kh_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        if self.Kc > 0:
            self.W_c = nn.Parameter(torch.Tensor(num_heads, Kc, embedding_dim))
            self.beta = nn.Linear(Kc, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, target_item, item_sequence, mask=None):
        """
        target_item: b x emb
        item_feat_seq: b x len x emb
        cross_feat_seq: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        batch_size = target_item.size(0)
        if self.Kc > 0:
            item_feat_seq, cross_feat_seq = torch.split(item_sequence,
                                                        [self.Kh_dim, self.Kc_dim], dim=-1)
            key_c = (cross_feat_seq.view(batch_size, self.Kc, -1).unsqueeze(1) 
                     * self.W_c.unsqueeze(0)).sum(-1) # b x h x Kc
            key_c_bias = self.beta(key_c) # b x h
        else:
            item_feat_seq = item_sequence
        
        # linear projection
        query = self.W_q(target_item)
        key_h = self.W_h(item_feat_seq)
        value = self.W_v(item_sequence)

        # split by heads
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_h = key_h.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attention
        scores = torch.matmul(query, key_h.transpose(-1, -2)) / self.scale # b x h x 1 x len
        if self.Kc > 0:
            scores += key_c_bias.view(batch_size, self.num_heads, 1, 1)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(batch_size, self.num_heads, 1, -1)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        topk = min(self.topk, scores.shape[-1]) # make sure input seq_len >= topk
        topk_scores, topk_index = scores.topk(topk, dim=-1, largest=True, sorted=True)
        # topk_value: b x h x topk x head_dim
        topk_value = torch.gather(value, 2, 
                                  topk_index.transpose(-1, -2).expand(-1, -1, -1, value.shape[-1]))
        attention = topk_scores.softmax(dim=-1) # b x h x 1 x topk
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, topk_value) # b x h x 1 x head_dim
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim) # b x attention_dim
        output = self.W_o(output)
        return output
