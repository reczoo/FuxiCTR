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
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MaskedSumPooling
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from fuxictr.utils import not_in_whitelist


class DIEN(BaseModel):
    """ Implementation of DIEN model based on the following reference code:
        https://github.com/mouna99/dien
    """
    def __init__(self,
                 feature_map,
                 model_id="DIEN",
                 gpu=-1,
                 dnn_hidden_units=[200, 80],
                 dnn_activations="ReLU",
                 learning_rate=1e-3,
                 embedding_dim=16,
                 net_dropout=0,
                 batch_norm=True,
                 gru_type="AUGRU",
                 enable_sum_pooling=False,
                 attention_dropout=0,
                 attention_type="bilinear_attention",
                 attention_hidden_units=[80, 40],
                 attention_activation="Dice",
                 use_attention_softmax=True,
                 item_info_fields=1,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DIEN, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.sum_pooling = MaskedSumPooling()
        self.gru_type = gru_type
        self.extraction_modules = nn.ModuleList()
        self.evolving_modules = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.extraction_modules.append(nn.GRU(input_size=self.item_info_dim, 
                                                hidden_size=self.item_info_dim, 
                                                batch_first=True))
        if gru_type in ["AGRU", "AUGRU"]:
            self.evolving_modules.append(DynamicGRU(self.item_info_dim,
                                                    self.item_info_dim, 
                                                    gru_type=gru_type))
        else:
            self.evolving_modules.append(nn.GRU(input_size=self.item_info_dim, 
                                                hidden_size=self.item_info_dim, 
                                                batch_first=True))
        if gru_type in ["AIGRU", "AGRU", "AUGRU"]:
            self.attention_modules.append(
                AttentionLayer(self.item_info_dim,
                                attention_type=attention_type,
                                attention_hidden_units=attention_hidden_units,
                                attention_activation=attention_activation,
                                use_attention_softmax=use_attention_softmax,
                                attention_dropout=attention_dropout))
        feature_dim = feature_map.sum_emb_out_dim() + self.item_info_dim * 3
        self.enable_sum_pooling = enable_sum_pooling
        if not self.enable_sum_pooling:
            feature_dim -= self.item_info_dim * 2
        self.dnn = MLP_Block(input_dim=feature_dim,
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
        batch_dict, item_dict, pad_mask = self.get_inputs(inputs)
        concat_emb = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            concat_emb.append(feature_emb)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = pad_mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        # remove rows without sequence elements
        non_zero_mask = pad_mask.sum(dim=1) > 0
        packed_interests, interest_emb = self.interest_extraction(0, sequence_emb[non_zero_mask], 
                                                                    pad_mask[non_zero_mask])
        h_out = self.interest_evolution(0, packed_interests, interest_emb, target_emb[non_zero_mask], 
                                        pad_mask[non_zero_mask])
        final_out = self.get_unmasked_tensor(h_out, non_zero_mask)
        concat_emb += [target_emb, final_out]
        if self.enable_sum_pooling: # sum pooling of behavior sequence is used in the paper code
            sum_pool_emb = self.sum_pooling(sequence_emb)
            concat_emb += [sum_pool_emb, target_emb * sum_pool_emb]
        y_pred = self.dnn(torch.cat(concat_emb, dim=-1))
        return_dict = {"y_pred": y_pred}
        return return_dict
  
    def get_unmasked_tensor(self, h, non_zero_mask):
        out = torch.zeros([non_zero_mask.size(0)] + list(h.shape[1:]), device=h.device)
        out[non_zero_mask] = h
        return out

    def interest_extraction(self, idx, sequence_emb, mask):
        seq_lens = mask.sum(dim=1).cpu()
        packed_seq = pack_padded_sequence(sequence_emb, 
                                          seq_lens, 
                                          batch_first=True, 
                                          enforce_sorted=False)
        packed_interests, _ = self.extraction_modules[idx](packed_seq)
        interest_emb, _ = pad_packed_sequence(packed_interests,
                                              batch_first=True,
                                              padding_value=0.0,
                                              total_length=mask.size(1))
        return packed_interests, interest_emb

    def interest_evolution(self, idx, packed_interests, interest_emb, target_emb, mask):
        if self.gru_type == "GRU":
            _, h_out = self.evolving_modules[idx](packed_interests)
        else:
            attn_scores = self.attention_modules[idx](interest_emb, target_emb, mask)
            seq_lens = mask.sum(dim=1).cpu()
            if self.gru_type == "AIGRU":
                packed_inputs = pack_padded_sequence(interest_emb * attn_scores,
                                                     seq_lens,
                                                     batch_first=True,
                                                     enforce_sorted=False)
                _, h_out = self.evolving_modules[idx](packed_inputs)
            else:
                packed_scores = pack_padded_sequence(attn_scores,
                                                     seq_lens,
                                                     batch_first=True,
                                                     enforce_sorted=False)
                _, h_out = self.evolving_modules[idx](packed_interests, packed_scores)
        return h_out.squeeze()

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


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, attention_type="bilinear_attention", attention_hidden_units=[80, 40],
                 attention_activation="Dice", use_attention_softmax=True, attention_dropout=0.0):
        super(AttentionLayer, self).__init__()
        assert attention_type in ["bilinear_attention", "dot_attention", "din_attention"], \
               "attention_type={} is not supported.".format(attention_type)
        self.attention_type = attention_type
        self.use_attention_softmax = use_attention_softmax
        if attention_type == "bilinear_attention":
            self.W_kernel = nn.Parameter(torch.eye(model_dim))
        elif attention_type == "din_attention":
            self.attn_mlp = MLP_Block(input_dim=model_dim * 4,
                                      output_dim=1,
                                      hidden_units=attention_hidden_units,
                                      hidden_activations=attention_activation,
                                      output_activation=None, 
                                      dropout_rates=attention_dropout,
                                      batch_norm=False)

    def forward(self, sequence_emb, target_emb, mask=None):
        seq_len = sequence_emb.size(1)
        if self.attention_type == "dot_attention":
            attn_score = sequence_emb @ target_emb.unsqueeze(-1)
        elif self.attention_type == "bilinear_attention":
            attn_score = (sequence_emb @ self.W_kernel) @ target_emb.unsqueeze(-1)
        elif self.attention_type == "din_attention":
            target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
            din_concat = torch.cat([target_emb, sequence_emb, target_emb - sequence_emb, 
                                    target_emb * sequence_emb], dim=-1)
            attn_score = self.attn_mlp(din_concat.view(-1, 4 * target_emb.size(-1)))
        attn_score = attn_score.view(-1, seq_len)
        if mask is not None:
            attn_score = attn_score * mask.float()
        if self.use_attention_softmax:
            if mask is not None:
                attn_score += -1.e9 * (1 - mask.float())
            attn_score = attn_score.softmax(dim=-1)
        return attn_score


class AGRUCell(nn.Module):
    r"""AGRUCell with attentional update gate
        Reference: GRUCell from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hx, attn):
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hx)
        
        i_u, i_r, i_n = gate_x.chunk(3, 1)
        h_u, h_r, h_n = gate_h.chunk(3, 1)
        
        reset_gate = torch.sigmoid(i_r + h_r)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = hx + attn.view(-1, 1) * (new_gate - hx)
        return hy


class AUGRUCell(nn.Module):
    r"""AUGRUCell with attentional update gate
        Reference: GRUCell from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hx, attn):
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hx)
        
        i_u, i_r, i_n = gate_x.chunk(3, 1)
        h_u, h_r, h_n = gate_h.chunk(3, 1)
        
        update_gate = torch.sigmoid(i_u + h_u)
        update_gate = update_gate * attn.unsqueeze(-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = hx + update_gate * (new_gate - hx)
        return hy


class DynamicGRU(nn.Module):
    r"""DynamicGRU with GRU, AIGRU, AGRU, and AUGRU choices
        Reference: https://github.com/GitHub-HongweiZhang/prediction-flow/blob/master/prediction_flow/pytorch/nn/rnn.py
    """
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AUGRU'):
        super(DynamicGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_type = gru_type
        if gru_type == "AUGRU":
            self.gru_cell = AUGRUCell(input_size, hidden_size, bias=bias)
        elif gru_type == "AGRU":
            self.gru_cell = AGRUCell(input_size, hidden_size, bias=bias)
    
    def forward(self, packed_seq_emb, attn_score=None, h=None):
        assert isinstance(packed_seq_emb, PackedSequence) and isinstance(attn_score, PackedSequence), \
               "DynamicGRU supports only `PackedSequence` input."
        x, batch_sizes, sorted_indices, unsorted_indices = packed_seq_emb
        attn, _, _, _ = attn_score

        if h == None:
            h = torch.zeros(batch_sizes[0], self.hidden_size, device=x.device)
        output_h = torch.zeros(batch_sizes[0], self.hidden_size, device=x.device)
        outputs = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        
        start = 0
        for batch_size in batch_sizes:
            _x = x[start: start + batch_size]
            _h = h[:batch_size]
            _attn = attn[start: start + batch_size]
            h = self.gru_cell(_x, _h, _attn)
            outputs[start: start + batch_size] = h
            output_h[:batch_size] = h
            start += batch_size
        
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices), \
               output_h[unsorted_indices]

