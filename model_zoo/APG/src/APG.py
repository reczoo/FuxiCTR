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
from fuxictr.pytorch.layers import MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class APG_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, condition_dim, bias=True, rank_k=None, 
                 overparam_p=None, generate_bias=False, hypernet_config={}):
        super(APG_Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generate_bias = generate_bias
        self.rank_k = rank_k
        self.use_low_rank = (rank_k is not None)
        self.use_over_param = (overparam_p is not None)
        self.use_bias = bias
        if self.use_low_rank:
            assert rank_k <= min(input_dim, output_dim), "Invalid rank_k={}".format(rank_k)
            if self.use_over_param:
                assert overparam_p >= rank_k, "Invalid overparam_p={}".format(overparam_p)
                self.U_l = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_dim, overparam_p)))
                self.U_r = nn.Parameter(nn.init.xavier_normal_(torch.empty(overparam_p, rank_k)))
                self.V_l = nn.Parameter(nn.init.xavier_normal_(torch.empty(rank_k, overparam_p)))
                self.V_r = nn.Parameter(nn.init.xavier_normal_(
                                        torch.empty(overparam_p, output_dim)))
            else:
                self.U = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_dim, rank_k)))
                self.V = nn.Parameter(nn.init.xavier_normal_(torch.empty(rank_k, output_dim)))
            # low-rank weight generation
            self.hypernet = MLP_Block(
                input_dim=condition_dim,
                output_dim=rank_k ** 2 + int(generate_bias) * output_dim,
                hidden_units=hypernet_config.get("hidden_units", []),
                hidden_activations=hypernet_config.get("hidden_activations", "ReLU"),
                output_activation=None,
                dropout_rates=hypernet_config.get("dropout_rates", 0),
                batch_norm=False)
        else:
            # full weight generation
            self.hypernet = MLP_Block(
                input_dim=condition_dim,
                output_dim=input_dim * output_dim + int(generate_bias) * output_dim,
                hidden_units=hypernet_config.get("hidden_units", []),
                hidden_activations=hypernet_config.get("hidden_activations", "ReLU"),
                output_activation=None,
                dropout_rates=hypernet_config.get("dropout_rates", 0),
                batch_norm=False)
        if self.use_bias and (not self.generate_bias):
            self.bias = nn.Parameter(torch.zeros(1, output_dim))
        else:
            self.bias = None

    def generate_weight(self, condition_z):
        weight_S = self.hypernet(condition_z)
        bias = self.bias
        if self.generate_bias:
            if self.use_bias:
                bias = weight_S[:, 0:self.output_dim]
            weight_S = weight_S[:, self.output_dim:]
        if self.use_low_rank:
            weight_S = weight_S.reshape(-1, self.rank_k, self.rank_k)
        else:
            weight_S = weight_S.reshape(-1, self.input_dim, self.output_dim)
        return weight_S, bias

    def forward(self, input_h, condition_z):
        weight_S, bias = self.generate_weight(condition_z)
        if self.use_low_rank:
            if self.use_over_param:
                self.U = torch.matmul(self.U_l, self.U_r) # input_dim x rank_k
                self.V = torch.matmul(self.V_l, self.V_r) # rank_k x output_dim
            h = torch.matmul(input_h, self.U) # b x rank_k
            h = torch.bmm(h.unsqueeze(1), weight_S).squeeze(1) # b x rank_k
            out = torch.matmul(h, self.V)
        else:
            out = torch.bmm(input_h.unsqueeze(1), weight_S).squeeze(1)
        if bias is not None:
            out += bias
        return out


class APG_MLP(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False, 
                 bn_only_once=False, # Set True for inference speed up
                 use_bias=True,
                 hypernet_config={},
                 condition_dim=None,
                 condition_mode="self-wise",
                 rank_k=None,
                 overparam_p=None,
                 generate_bias=True):
        super(APG_MLP, self).__init__()
        self.hidden_layers = len(hidden_units)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * self.hidden_layers
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * self.hidden_layers
        hidden_activations = get_activation(hidden_activations, hidden_units)
        if not isinstance(rank_k, list):
            rank_k = [rank_k] * self.hidden_layers
        if not isinstance(overparam_p, list):
            overparam_p = [overparam_p] * self.hidden_layers
        assert self.hidden_layers == len(dropout_rates) == len(hidden_activations) \
               == len(rank_k) == len(overparam_p)
        hidden_units = [input_dim] + hidden_units
        self.dense_layers = nn.ModuleDict()
        if batch_norm and bn_only_once:
            self.dense_layers["bn_0"] = nn.BatchNorm1d(input_dim)
        self.condition_mode = condition_mode
        assert condition_mode in ["self-wise", "group-wise", "mix-wise"], \
               "Invalid condition_mode={}".format(condition_mode)
        for idx in range(self.hidden_layers):
            if self.condition_mode == "self-wise":
                condition_dim = hidden_units[idx]
            self.dense_layers["linear_{}".format(idx + 1)] = APG_Linear(
                hidden_units[idx], 
                hidden_units[idx + 1],
                condition_dim,
                bias=use_bias,
                rank_k=rank_k[idx],
                overparam_p=overparam_p[idx],
                generate_bias=generate_bias,
                hypernet_config=hypernet_config)
            if batch_norm and not bn_only_once:
                self.dense_layers["bn_{}".format(idx + 1)] = nn.BatchNorm1d(hidden_units[idx + 1])
            if hidden_activations[idx]:
                self.dense_layers["act_{}".format(idx + 1)] = hidden_activations[idx]
            if dropout_rates[idx] > 0:
                self.dense_layers["drop_{}".format(idx + 1)] = nn.Dropout(p=dropout_rates[idx])
        if output_dim is not None:
            self.dense_layers["out_proj"] = nn.Linear(hidden_units[-1], output_dim, bias=use_bias)
        if output_activation is not None:
            self.dense_layers["out_act"] = get_activation(output_activation)
    
    def forward(self, x, condition_z=None):
        if "bn_0" in self.dense_layers:
            x = self.dense_layers["bn_0"](x)
        for idx in range(self.hidden_layers):
            if self.condition_mode == "self-wise":
                x = self.dense_layers["linear_{}".format(idx + 1)](x, x)
            else:
                x = self.dense_layers["linear_{}".format(idx + 1)](x, condition_z)
            if "bn_{}".format(idx + 1) in self.dense_layers:
                x = self.dense_layers["bn_{}".format(idx + 1)](x)
            if "act_{}".format(idx + 1) in self.dense_layers:
                x = self.dense_layers["act_{}".format(idx + 1)](x)
            if "drop_{}".format(idx + 1) in self.dense_layers:
                x = self.dense_layers["drop_{}".format(idx + 1)](x)
        if "out_proj" in self.dense_layers:
            x = self.dense_layers["out_proj"](x)
        if "out_act" in self.dense_layers:
            x = self.dense_layers["out_act"](x)
        return x
