# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, \
                                   MaskedSumPooling


class DMR(BaseModel):
    """ Implementation of DMR model based on the following reference code:
        https://github.com/lvze92/DMR
        https://github.com/thinkall/Contrib/tree/master/DMR
    """
    def __init__(self,
                 feature_map,
                 model_id="DMR",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 net_dropout=0,
                 batch_norm=True,
                 bn_only_once=False,
                 target_field=("item_id", "cate_id"),
                 sequence_field=("click_history", "cate_history"),
                 neg_seq_field=("neg_click_history", "neg_cate_history"),
                 context_field="btag",
                 enable_sum_pooling=False,
                 enable_u2i_rel=True,
                 enable_i2i_rel=False,
                 attention_hidden_units=[80, 40],
                 attention_activation="ReLU",
                 attention_dropout=0,
                 use_pos_emb=True,
                 pos_emb_dim=8,
                 aux_loss_beta=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DMR, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if target_field and not isinstance(target_field, list):
            target_field = [target_field]
        self.target_field = target_field
        if sequence_field and not isinstance(sequence_field, list):
            sequence_field = [sequence_field]
        self.sequence_field = sequence_field
        if neg_seq_field and not isinstance(neg_seq_field, list):
            neg_seq_field = [neg_seq_field]
        self.neg_seq_field = neg_seq_field
        if context_field and not isinstance(context_field, list):
            context_field = [context_field]
        self.context_field = context_field
        assert len(target_field) == len(sequence_field)
        if neg_seq_field:
            assert len(neg_seq_field) == len(sequence_field)
        if context_field:
            assert len(context_field) == len(sequence_field)
        self.aux_loss_beta = aux_loss_beta
        self.enable_sum_pooling = enable_sum_pooling
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim,
            not_required_feature_columns=flatten([self.neg_seq_field]) if self.neg_seq_field else None)
        self.sum_pooling = MaskedSumPooling()
        self.output_emb_layer = nn.ModuleDict() # output vocab embedding
        for feature in flatten([self.target_field]):
            feature_spec = feature_map.features[feature]
            self.output_emb_layer[feature] = nn.Embedding(feature_spec["vocab_size"], 
                                                          embedding_dim, 
                                                          padding_idx=feature_spec["padding_idx"])
        if self.context_field is not None:
            self.context_emb_layer = nn.ModuleDict() # context field embedding
            for feature in flatten([self.context_field]):
                feature_spec = feature_map.features[feature]
                self.context_emb_layer[feature] = nn.Embedding(feature_spec["vocab_size"], 
                                                               embedding_dim,
                                                               padding_idx=feature_spec["padding_idx"])
        self.enable_u2i_rel = enable_u2i_rel
        self.enable_i2i_rel = enable_i2i_rel
        self.u2i_net = nn.ModuleList()
        self.i2i_net = nn.ModuleList()
        feature_dim = feature_map.sum_emb_out_dim()
        for i in range(len(self.target_field)):
            model_dim = embedding_dim * len(list(flatten([self.target_field[i]])))
            max_seq_len = feature_map.features[list(flatten([self.sequence_field[i]]))[0]]["max_len"]
            if self.enable_sum_pooling:
                feature_dim += model_dim * 2
            if self.context_field:
                context_dim = embedding_dim * len(list(flatten([self.context_field[i]])))
            else:
                context_dim = 0
            if enable_u2i_rel:
                self.u2i_net.append(User2ItemNet(context_dim,
                                                 model_dim,
                                                 attention_hidden_units=attention_hidden_units, 
                                                 attention_activation=attention_activation, 
                                                 attention_dropout=attention_dropout, 
                                                 pos_emb_dim=pos_emb_dim, 
                                                 max_seq_len=max_seq_len))
                feature_dim += 1
            if enable_i2i_rel:
                feature_dim += 1
            self.i2i_net.append(Item2ItemNet(context_dim, model_dim, 
                                             attention_hidden_units=attention_hidden_units, 
                                             attention_activation=attention_activation,
                                             attention_dropout=attention_dropout, 
                                             use_pos_emb=use_pos_emb,
                                             pos_emb_dim=pos_emb_dim, 
                                             max_seq_len=max_seq_len))
        if self.neg_seq_field is not None:
            feature_dim -= embedding_dim * len(set(flatten([self.neg_seq_field])))
        self.dnn = MLP_Block(input_dim=feature_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             bn_only_once=bn_only_once)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        concat_emb = []
        aux_loss_sum = 0
        for i in range(len(self.target_field)):
            target_emb = self.get_embedding(self.target_field[i], feature_emb_dict)
            sequence_emb = self.get_embedding(self.sequence_field[i], feature_emb_dict)
            seq_field = list(flatten([self.sequence_field[i]]))[0] # pick the first sequence field
            pad_mask = X[seq_field].long() > 0
            # item2item network
            context_emb = self.get_embedding(self.context_field[i], feature_emb_dict) \
                          if self.context_field else None
            attn_out, rel_i2i = self.i2i_net[i](target_emb, sequence_emb, context_emb, mask=pad_mask)
            concat_emb.append(attn_out)
            if self.enable_i2i_rel:
                concat_emb.append(rel_i2i)
            if self.enable_u2i_rel:
                # user2item network
                neg_emb = self.get_out_embedding(self.neg_seq_field[i], self.target_field[i], X) \
                          if self.aux_loss_beta > 0 else None
                target_emb2 = self.get_out_embedding(self.target_field[i], self.target_field[i], X)
                sequence_emb2 = self.get_out_embedding(self.sequence_field[i], self.target_field[i], X)
                context_emb2 = self.get_context_embedding(self.context_field[i], X) if self.context_field else None
                rel_u2i, aux_loss = self.u2i_net[i](target_emb2, sequence_emb, context_emb2, 
                                                    sequence_emb2, neg_emb, mask=pad_mask)
                aux_loss_sum += aux_loss
                concat_emb.append(rel_u2i)
            if self.enable_sum_pooling: # sum pooling of behavior sequence is used in the paper code
                sum_pool_emb = self.sum_pooling(sequence_emb)
                concat_emb += [sum_pool_emb, target_emb * sum_pool_emb]
        for feature, emb in feature_emb_dict.items():
            if emb.ndim == 2 and (feature not in set(flatten([self.neg_seq_field]))):
                concat_emb.append(emb)
        y_pred = self.dnn(torch.cat(concat_emb, dim=-1))
        return_dict = {"y_pred": y_pred, "aux_loss": aux_loss_sum}
        return return_dict

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        if self.aux_loss_beta > 0:
            # padding post required
            loss += self.aux_loss_beta * return_dict["aux_loss"]
        return loss

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_out_embedding(self, field, target_field, X):
        emb_list = []
        for input_name, emb_name in zip(flatten([field]), flatten([target_field])):
            emb = self.output_emb_layer[emb_name](X[input_name].long())
            emb_list.append(emb)
        return torch.cat(emb_list, dim=-1)

    def get_context_embedding(self, field, X):
        emb_list = []
        for feature in zip(flatten([field])):
            emb = self.context_emb_layer[feature](X[feature].long())
            emb_list.append(emb)
        return torch.cat(emb_list, dim=-1)


class User2ItemNet(nn.Module):
    def __init__(self, context_dim=64, model_dim=64, attention_hidden_units=[80, 40], 
                 attention_activation="ReLU", attention_dropout=0.0, pos_emb_dim=8, 
                 max_seq_len=50):
        """ We follow the code from the authors for this implementation.
        """
        super(User2ItemNet, self).__init__()
        self.model_dim = model_dim
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, pos_emb_dim))
        self.context_dim = context_dim + pos_emb_dim
        self.W_q = nn.Sequential(nn.Linear(self.context_dim, model_dim),
                                 nn.ReLU())
        self.attn_mlp = MLP_Block(input_dim=model_dim * 4,
                                  output_dim=1,
                                  hidden_units=attention_hidden_units,
                                  hidden_activations=attention_activation,
                                  output_activation=None, 
                                  dropout_rates=attention_dropout,
                                  batch_norm=False)
        self.W_o = nn.Sequential(nn.Linear(model_dim, model_dim),
                                 nn.ReLU())

    def forward(self, target_emb, sequence_emb, context_emb, sequence_emb2, neg_emb=None, mask=None):
        batch_size = target_emb.size(0)
        if context_emb is None:
            context_emb = self.pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            context_emb = torch.cat([self.pos_emb.unsqueeze(0).expand(batch_size, -1, -1),
                                     context_emb], dimi=-1)
        seq_len = sequence_emb.size(1)
        query = self.W_q(context_emb.reshape(-1, self.context_dim)).reshape(-1, seq_len, self.model_dim)
        inp_concat = torch.cat([query, sequence_emb, query - sequence_emb, 
                                query * sequence_emb], dim=-1)
        attn_score = self.attn_mlp(inp_concat.view(-1, 4 * self.model_dim))
        attn_score = attn_score.view(-1, seq_len) # b x len
        attn_mask = self.get_mask(mask) # 0 for masked positions
        expand_score = attn_score.unsqueeze(1).repeat(1, seq_len, 1) # b x len x len
        expand_score = expand_score.masked_fill_(attn_mask == False, -1.e9) # fill -inf if mask=False
        expand_score = expand_score.softmax(dim=-1)
        user_embs = torch.bmm(expand_score, sequence_emb) # b x len x d
        user_embs = self.W_o(user_embs.reshape(-1, self.model_dim)).reshape(-1, seq_len, self.model_dim)
        rel_u2i = torch.sum(user_embs[:, -1, :] * target_emb, dim=-1, keepdim=True)
        if neg_emb is not None:
            pos_prob = torch.sum(user_embs[:, -2, :] * sequence_emb2[:, -1, :], dim=-1).sigmoid()
            neg_prob = torch.sum(user_embs[:, -2, :] * neg_emb, dim=-1).sigmoid()
            aux_loss = -torch.log(pos_prob) - torch.log(1 - neg_prob)
            aux_loss = (aux_loss * mask[:, -1]).sum() / mask[:, -1].sum()
        else:
            aux_loss = 0
        return rel_u2i, aux_loss

    def get_mask(self, mask):
        """ attn_mask: B x L, 0 for masked positions
        """
        seq_len = mask.size(1)
        attn_mask = mask.unsqueeze(1).repeat(1, seq_len, 1) # B x L x L
        causal_mask = (torch.tril(torch.ones(seq_len, seq_len, device=mask.device)).bool()
                       .unsqueeze(0).expand_as(attn_mask))
        attn_mask = attn_mask & causal_mask
        diag_ones = torch.eye(seq_len, device=mask.device).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask | diag_ones
        return attn_mask


class Item2ItemNet(nn.Module):
    def __init__(self, context_dim=64, model_dim=64, attention_hidden_units=[80, 40], 
                 attention_activation="ReLU", attention_dropout=0.0, use_pos_emb=True,
                 pos_emb_dim=8, max_seq_len=50):
        super(Item2ItemNet, self).__init__()
        self.model_dim = model_dim
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, pos_emb_dim))
            context_dim += pos_emb_dim
        self.context_dim = context_dim + model_dim
        self.W_q = nn.Sequential(nn.Linear(self.context_dim, model_dim),
                                 nn.ReLU())
        self.attn_mlp = MLP_Block(input_dim=model_dim * 4,
                                  output_dim=1,
                                  hidden_units=attention_hidden_units,
                                  hidden_activations=attention_activation,
                                  output_activation=None,
                                  dropout_rates=attention_dropout,
                                  batch_norm=False)

    def forward(self, target_emb, sequence_emb, context_emb=None, mask=None):
        seq_len = sequence_emb.size(1)
        if context_emb is None:
            context_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            context_emb = torch.cat([target_emb.unsqueeze(1).expand(-1, seq_len, -1),
                                     context_emb], dimi=-1)
        if self.use_pos_emb:
            context_emb = torch.cat([context_emb,
                                     self.pos_emb.unsqueeze(0).expand(context_emb.size(0), -1, -1)], 
                                     dim=-1)
        query = self.W_q(context_emb.reshape(-1, self.context_dim)).view(-1, seq_len, self.model_dim)
        inp_concat = torch.cat([query, sequence_emb, query - sequence_emb, 
                                query * sequence_emb], dim=-1)
        attn_score = self.attn_mlp(inp_concat.view(-1, 4 * self.model_dim))
        attn_score = attn_score.view(-1, seq_len) # b x len
        score_softmax = attn_score.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        score_softmax = score_softmax.softmax(dim=-1)
        attn_out = (score_softmax.unsqueeze(-1) * sequence_emb).sum(dim=1)
        scores_no_softmax = attn_score * mask.float() # fill 0 if mask=0
        rel_i2i = scores_no_softmax.sum(dim=1, keepdim=True)
        return attn_out, rel_i2i

