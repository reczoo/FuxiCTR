# =========================================================================
# Copyright (C) 2025. salmon1802@github.
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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation
import torch.nn.functional as F


class QNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="QNN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 hidden_activations='ReLU',
                 neuron_type='T1',
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(QNN, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.qnn = QuadraticNeuralNetworks(input_dim=input_dim,
                                           num_cross_layers=num_cross_layers,
                                           net_dropout=net_dropout,
                                           hidden_activations=hidden_activations,
                                           neuron_type=neuron_type,
                                           batch_norm=batch_norm)
        self.neuron_type = neuron_type
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)
        y_pred = self.qnn(feature_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class QuadraticNeuralNetworks(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 net_dropout=0.1,
                 hidden_activations='relu',
                 neuron_type='T1',
                 batch_norm=False):
        super(QuadraticNeuralNetworks, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.dropout = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.layer = nn.ModuleList()
        self.activation = nn.ModuleList()
        self.neuron_type = neuron_type
        if neuron_type in ["T1", "T2", "T6"]:
            self.compressed = nn.Linear(input_dim, 100, bias=False)
            self.fc = nn.Linear(100, 1)
        else:
            self.fc = nn.Linear(input_dim, 1)
        for i in range(num_cross_layers):
            self.layer.append(QuadraticLayer(input_dim, neuron_type=neuron_type))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(input_dim))
            self.activation.append(get_activation(hidden_activations))

    def forward(self, x):
        if self.neuron_type in ["T1", "T2", "T6"]:
            x = self.compressed(x)
        for i in range(self.num_cross_layers):
            x = self.layer[i](x)
            if len(self.norm) > i:
                x = self.norm[i](x)
            if self.activation[i] is not None:
                x = self.activation[i](x)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.fc(x)
        return logit

class QuadraticLayer(nn.Module):
    def __init__(self, input_dim, neuron_type="T1"):
        super(QuadraticLayer, self).__init__()
        self.neuron_type = neuron_type
        self.input_dim = input_dim
        if neuron_type in ["T1", "T6"]:
            self.bi_linear = nn.Bilinear(100, 100, 100)
            self.linear = nn.Linear(100, 100)
        elif neuron_type == "T2":
            self.bi_linear = nn.Bilinear(100, 100, 100)
        elif neuron_type in ["T3", "T4", "T9", "T11", "T14", "T19", "T20", "T21"]:
            self.linear = nn.Linear(input_dim, input_dim)
        elif neuron_type in ["T5", "T10", "T16", "T17", "T23", "T24"]:
            self.linear = nn.Linear(input_dim, input_dim * 2)
        elif neuron_type == "T7":
            self.linear1 = nn.Linear(input_dim, input_dim * 2)
            self.linear2 = nn.Linear(input_dim, input_dim)
        elif neuron_type == "T8":
            self.linear = nn.Linear(input_dim, input_dim * 3)
        elif neuron_type == "T12":
            self.linear1 = nn.Linear(input_dim, input_dim)
            self.linear2 = nn.Linear(input_dim, input_dim // 2)
        elif neuron_type == "T13":
            self.linear = nn.Linear(input_dim, input_dim // 2 * 3)
        elif neuron_type == "T15":
            self.linear = nn.Linear(input_dim, input_dim // 2)
        elif neuron_type == "T18":
            self.linear = nn.Linear(input_dim, input_dim)
            self.downsize = nn.Linear(input_dim * 2, input_dim)
        elif neuron_type == "T22":
            self.linear = nn.Linear(input_dim, input_dim)
            self.alpha = nn.Parameter(torch.ones(input_dim))  # 可学习的权重
        elif neuron_type == "T25":
            self.linear = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
                                        nn.ReLU(),
                                        nn.Linear(input_dim // 2, input_dim))
        elif neuron_type in ["T26", "T27", "T28", "T29", "T30"]:
            self.linear = nn.Linear(input_dim, input_dim * 2)
        else:
            assert "there is no such neuron_type type!"

    def forward(self, x):
        if self.neuron_type == "T1":
            x = self.bi_linear(x, x) + self.linear(x)
        elif self.neuron_type == "T2":
            x = self.bi_linear(x, x)
        elif self.neuron_type == "T3":
            x = self.linear(x * x)
        elif self.neuron_type == "T4":
            x = self.linear(x)
            x = x * x
        elif self.neuron_type == "T5":
            x = self.linear(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=-1)
            x = x1 * x2
        elif self.neuron_type == "T6":
            x = self.bi_linear(x, x) + self.linear(x * x)
        elif self.neuron_type == "T7":
            h = self.linear1(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = h1 * h2 + self.linear2(x * x)
        elif self.neuron_type == "T8":
            x = self.linear(x)
            x1, x2, x3 = torch.chunk(x, chunks=3, dim=-1)
            x = x1 * x2 + x3
        elif self.neuron_type == "T9":
            x = x * self.linear(x) + x
        elif self.neuron_type == "T10":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = x * h1 + h2
        elif self.neuron_type == "T11":
            x = self.linear(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=-1)
            x = torch.cat([x1, x2 * x2], dim=-1)
        elif self.neuron_type == "T12":
            h = self.linear1(x)
            x = self.linear2(x * x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = torch.cat([h1 * h2, x], dim=-1)
        elif self.neuron_type == "T13":
            x = self.linear(x)
            x1, x2, x3 = torch.chunk(x, chunks=3, dim=-1)
            x = torch.cat([x1 * x2, x3], dim=-1)
        elif self.neuron_type == "T14":
            x = self.linear(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=-1)
            x = torch.cat([x1 * x2, x2], dim=-1)
        elif self.neuron_type == "T15":
            x = self.linear(x)
            x = torch.cat([x * x, x], dim=-1)
        # T9 variants
        elif self.neuron_type == "T16": # 双线性
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = x * (h1 * h2) + x
        elif self.neuron_type == "T17":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = x * h1 + h2 + x
        elif self.neuron_type == "T18":
            x = torch.cat([x * self.linear(x), x], dim=-1)
            x = self.downsize(x)
        elif self.neuron_type == "T19":
            x = F.relu(self.linear(x)) * x + x
        elif self.neuron_type == "T20":
            h = self.linear(x)
            x = x * h + h + x
        elif self.neuron_type == "T21":
            x = (x * self.linear(x)) ** 2 + x
        elif self.neuron_type == "T22":
            x = x * self.linear(x) + self.alpha * x
        elif self.neuron_type == "T23":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = h1 * h2 + x
        elif self.neuron_type == "T24":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = h1 * F.relu(h2) + x
        elif self.neuron_type == "T25":
            x = x * self.linear(x) + x
        elif self.neuron_type == "T26":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = torch.stack([F.relu(h1) * x, F.relu(h2) * x], dim=1).mean(dim=1) + x
        elif self.neuron_type == "T27":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = torch.stack([F.relu(h1) * (x**2), F.relu(h2) * (x**2)], dim=1).mean(dim=1) + x
        elif self.neuron_type == "T28":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = torch.stack([(F.relu(h1)**2) * x, (F.relu(h2)**2) * x], dim=1).mean(dim=1) + x
        elif self.neuron_type == "T29":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = torch.stack([(F.relu(h1)) * x, (F.relu(h2)**2) * x], dim=1).mean(dim=1) + x
        elif self.neuron_type == "T30":
            h = self.linear(x)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            x = torch.stack([F.relu(h1) * x, F.relu(h2) * (x**2)], dim=1).mean(dim=1) + x
        return x
