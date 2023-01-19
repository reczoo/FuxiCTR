# =========================================================================
# Copyright (C) 2023. The FuxiCTR Authors. All rights reserved.
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
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, ScaledDotProductAttention, \
                                   MaskedSumPooling
from torch.nn import MultiheadAttention


class DMIN(BaseModel):
    """ Implementation of DMIN model based on the reference code:
        https://github.com/mengxiaozhibo/DMIN
    """
    def __init__(self,
                 feature_map,
                 model_id="DMIN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="Dice",
                 aux_hidden_units=[100, 50],
                 aux_activation="Sigmoid",
                 net_dropout=0,
                 target_field=("item_id", "cate_id"),
                 sequence_field=("click_history", "cate_history"),
                 neg_seq_field=("neg_click_history", "neg_cate_history"),
                 num_heads=4,
                 attention_hidden_units=[80, 40],
                 attention_activation="Sigmoid",
                 attention_dropout=0,
                 use_pos_emb=True,
                 pos_emb_dim=2,
                 aux_loss_lambda=0,
                 batch_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DMIN, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        for x in [target_field, sequence_field, neg_seq_field]:
            if x != None and type(x) not in [str, tuple]:
                raise ValueError("{} should be either str or tuple.".format(x))
        self.target_field = target_field
        self.sequence_field = sequence_field
        self.aux_loss_lambda = aux_loss_lambda
        self.neg_seq_field = neg_seq_field
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.sum_pooling = MaskedSumPooling()
        model_dim = embedding_dim * len(self.target_field) if type(self.target_field) == tuple \
                    else embedding_dim
        max_seq_len = feature_map.features[self.sequence_field[0]]["max_len"] \
                      if type(self.target_field) == tuple else \
                      feature_map.features[self.sequence_field]["max_len"]
        self.behavior_refiner = BehaviorRefinerLayer(model_dim, 
                                                     ffn_dim=model_dim * 2, 
                                                     num_heads=num_heads,
                                                     attn_dropout=attention_dropout,
                                                     net_dropout=net_dropout)
        self.multi_interest_extractor = MultiInterestExtractorLayer(model_dim,
                                                                    ffn_dim=model_dim * 2, 
                                                                    num_heads=num_heads,
                                                                    attn_dropout=attention_dropout,
                                                                    net_dropout=net_dropout,
                                                                    attn_hidden_units=attention_hidden_units,
                                                                    attn_activation=attention_activation,
                                                                    use_pos_emb=use_pos_emb,
                                                                    pos_emb_dim=pos_emb_dim,
                                                                    max_seq_len=max_seq_len)
        feature_dim = feature_map.sum_emb_out_dim() + model_dim * (num_heads + 1)
        if self.neg_seq_field is not None:
            feature_dim -= embedding_dim * len(list(flatten([self.neg_seq_field])))
        self.dnn = nn.Sequential(nn.BatchNorm1d(feature_dim) if batch_norm else nn.Identity(),
                                 MLP_Block(input_dim=feature_dim,
                                           output_dim=1,
                                           hidden_units=dnn_hidden_units,
                                           hidden_activations=dnn_activations,
                                           output_activation=self.output_activation,
                                           dropout_rates=net_dropout))
        if self.aux_loss_lambda > 0:
            self.model_dim = model_dim
            self.aux_net = nn.Sequential(nn.BatchNorm1d(model_dim * 2) if batch_norm else nn.Identity(),
                                         MLP_Block(input_dim=model_dim * 2,
                                                   output_dim=1,
                                                   hidden_units=aux_hidden_units,
                                                   hidden_activations=aux_activation,
                                                   output_activation="sigmoid",
                                                   dropout_rates=net_dropout))
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        target_emb = self.get_embedding(self.target_field, feature_emb_dict)
        sequence_emb = self.get_embedding(self.sequence_field, feature_emb_dict)
        sum_pool_emb = self.sum_pooling(sequence_emb)
        neg_emb = self.get_embedding(self.neg_seq_field, feature_emb_dict) \
                  if self.aux_loss_lambda > 0 else None
        seq_field = list(flatten([self.sequence_field]))[0] # pick the first sequence field
        pad_mask, attn_mask = self.get_mask(X[seq_field])
        refined_sequence = self.behavior_refiner(sequence_emb, attn_mask=attn_mask)
        interests = self.multi_interest_extractor(refined_sequence, target_emb, 
                                                  attn_mask=attn_mask, pad_mask=pad_mask)
        concat_emb = [sum_pool_emb, target_emb * sum_pool_emb] + interests
        for feature, emb in feature_emb_dict.items():
            if emb.ndim == 2 and (feature not in flatten([self.neg_seq_field])):
                concat_emb.append(emb)
        y_pred = self.dnn(torch.cat(concat_emb, dim=-1))
        return_dict = {"y_pred": y_pred, "head_emb": refined_sequence, "pos_emb": sequence_emb,
                       "neg_emb": neg_emb, "pad_mask": pad_mask}
        return return_dict

    def get_mask(self, x):
        """ padding_mask: 0 for masked positions
            attn_mask: 0 for masked positions
        """
        padding_mask = (x > 0)
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len * self.num_heads, 1).view(-1, seq_len, seq_len)
        diag_ones = torch.eye(seq_len, device=x.device).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask | diag_ones
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool() \
                           .unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & causal_mask
        return padding_mask, attn_mask

    def add_loss(self, inputs):
        y_true = self.get_labels(inputs)
        return_dict = self.forward(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        if self.aux_loss_lambda > 0:
            # padding post required
            head_emb, pos_emb, neg_emb, pad_mask = return_dict["head_emb"], \
                                                   return_dict["pos_emb"], \
                                                   return_dict["neg_emb"], \
                                                   return_dict["pad_mask"]
            pos_prob = self.aux_net(torch.cat([head_emb[:, :-1, :], pos_emb[:, 1:, :]], 
                                    dim=-1).view(-1, self.model_dim * 2))
            neg_prob = self.aux_net(torch.cat([head_emb[:, :-1, :], neg_emb[:, 1:, :]], 
                                    dim=-1).view(-1, self.model_dim * 2))
            aux_prob = torch.cat([pos_prob, neg_prob], dim=0).view(-1, 1)
            aux_label = torch.cat([torch.ones_like(pos_prob, device=aux_prob.device),
                                   torch.zeros_like(neg_prob, device=aux_prob.device)], dim=0).view(-1, 1)
            aux_loss = F.binary_cross_entropy(aux_prob, aux_label, reduction='none')
            pad_mask = pad_mask[:, 1:].view(-1, 1)
            aux_loss = torch.sum(aux_loss * pad_mask, dim=-1) / (torch.sum(pad_mask, dim=-1) + 1.e-9)
            loss += self.aux_loss_lambda * aux_loss
        return loss

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]


class BehaviorRefinerLayer(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=4, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super(BehaviorRefinerLayer, self).__init__()
        self.attention = MultiheadAttention(model_dim,
                                            num_heads=num_heads, 
                                            dropout=attn_dropout,
                                            batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                 nn.ReLU(),
                                 nn.Linear(ffn_dim, model_dim))
        self.use_residual = use_residual
        self.dropout = nn.Dropout(net_dropout)
        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm else None

    def forward(self, x, attn_mask=None):
        attn_mask = 1 - attn_mask.float() # 1 for masked positions in nn.MultiheadAttention
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        s = self.dropout(attn)
        if self.use_residual:
            s += x
        if self.layer_norm is not None:
            s = self.layer_norm(s)
        out = self.ffn(s)
        if self.use_residual:
            out += s
        return out


class MultiInterestExtractorLayer(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=4, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True, attn_hidden_units=[80, 40], attn_activation="Sigmoid",
                 use_pos_emb=True, pos_emb_dim=2, max_seq_len=10):
        super(MultiInterestExtractorLayer, self).__init__()
        assert model_dim % num_heads == 0, \
               "model_dim={} is not divisible by num_heads={}".format(model_dim, num_heads)
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5
        self.W_qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention = ScaledDotProductAttention(attn_dropout)
        self.W_o = nn.ModuleList([nn.Linear(self.head_dim, model_dim, bias=False) for _ in range(num_heads)])
        self.dropout = nn.ModuleList([nn.Dropout(net_dropout) for _ in range(num_heads)]) \
                       if net_dropout > 0 else None
        self.layer_norm = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_heads)]) \
                          if layer_norm else None
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                                nn.ReLU(),
                                                nn.Linear(ffn_dim, model_dim)) \
                                  for _ in range(num_heads)])
        self.TargetAttention = nn.ModuleList([TargetAttention(model_dim,
                                                              attention_hidden_units=attn_hidden_units,
                                                              attention_activation=attn_activation,
                                                              attention_dropout=attn_dropout,
                                                              use_pos_emb=use_pos_emb,
                                                              pos_emb_dim=pos_emb_dim,
                                                              max_seq_len=max_seq_len) \
                                            for _ in range(num_heads)])

    def forward(self, sequence_emb, target_emb, attn_mask=None, pad_mask=None):
        # linear projection
        query, key, value = torch.chunk(self.W_qkv(sequence_emb), chunks=3, dim=-1)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        attn, _ = self.attention(query, key, value, scale=self.scale, mask=attn_mask)
        # split heads
        attn_heads = torch.chunk(attn, chunks=self.num_heads, dim=1)
        interests = []
        for idx, h_head in enumerate(attn_heads):
            s = self.W_o[idx](h_head.squeeze(1))
            if self.dropout is not None:
                s = self.dropout[idx](s)
            if self.use_residual:
                s += sequence_emb
            if self.layer_norm is not None:
                s = self.layer_norm[idx](s)
            head_out = self.ffn[idx](s)
            if self.use_residual:
                head_out += s
            interest_emb = self.TargetAttention[idx](head_out, target_emb, mask=pad_mask)
            interests.append(interest_emb)
        return interests


class TargetAttention(nn.Module):
    def __init__(self, 
                 model_dim=64,
                 attention_hidden_units=[80, 40], 
                 attention_activation="Sigmoid",
                 attention_dropout=0,
                 use_pos_emb=True,
                 pos_emb_dim=2,
                 max_seq_len=10):
        super(TargetAttention, self).__init__()
        self.model_dim = model_dim
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, pos_emb_dim))
            self.W_proj = nn.Linear(model_dim + pos_emb_dim, model_dim)
        self.attn_mlp = MLP_Block(input_dim=model_dim * 4,
                                  output_dim=1,
                                  hidden_units=attention_hidden_units,
                                  hidden_activations=attention_activation,
                                  output_activation=None, 
                                  dropout_rates=attention_dropout,
                                  batch_norm=False)

    def forward(self, sequence_emb, target_emb, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        seq_len = sequence_emb.size(1)
        target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        if self.use_pos_emb:
            target_emb = torch.cat([target_emb, self.pos_emb.expand(target_emb.size(0), -1, -1)], dim=-1)
            target_emb = self.W_proj(target_emb)
        din_concat = torch.cat([target_emb, sequence_emb, target_emb - sequence_emb, 
                                target_emb * sequence_emb], dim=-1)
        attn_score = self.attn_mlp(din_concat.view(-1, 4 * target_emb.size(-1)))
        attn_score = attn_score.view(-1, seq_len) # b x len
        if mask is not None:
            attn_score = attn_score.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
            attn_score = attn_score.softmax(dim=-1)
        output = (attn_score.unsqueeze(-1) * sequence_emb).sum(dim=1)
        return output
