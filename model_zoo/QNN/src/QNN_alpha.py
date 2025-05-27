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

class QNN_alpha(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="QNN_alpha",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 num_layers=3,
                 num_heads=1,
                 num_row=2,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(QNN_alpha, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        num_fields = feature_map.num_fields
        self.qnn = QuadraticNeuralNetworks(input_dim=input_dim,
                                           num_layers=num_layers,
                                           net_dropout=net_dropout,
                                           num_heads=num_heads,
                                           num_row=num_row,
                                           batch_norm=batch_norm,
                                           num_fields=num_fields)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)
        if self.training:
            y_pred1 = self.qnn(feature_emb)
            y_pred2 = self.qnn(feature_emb)
            return_dict = {"y_pred1": self.output_activation(y_pred1),
                           "y_pred2": self.output_activation(y_pred2)}
        else:
            y_pred = self.qnn(feature_emb)
            y_pred = self.output_activation(y_pred)
            return_dict = {"y_pred": y_pred}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred1 = return_dict["y_pred1"]
        y_pred2 = return_dict["y_pred2"]
        y_pred = (y_pred1 + y_pred2) * 0.5
        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss1 = self.loss_fn(y_pred1, y_pred.detach(), reduction='mean')
        loss2 = self.loss_fn(y_pred2, y_pred.detach(), reduction='mean')
        loss = loss + loss1 + loss2
        return loss

class QuadraticNeuralNetworks(nn.Module):
    def __init__(self,
                 input_dim,
                 num_fields,
                 num_layers=3,
                 net_dropout=0.1,
                 num_heads=2,
                 num_row=2,
                 batch_norm=False):
        super(QuadraticNeuralNetworks, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.ModuleList()
        self.layer = nn.ModuleList()
        for i in range(num_layers):
            self.layer.append(QuadraticLayer(input_dim, num_row=num_row, num_heads=num_heads, batch_norm=batch_norm, num_fields=num_fields, net_dropout=net_dropout))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layer[i](x)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.fc(x)
        return logit


class QuadraticLayer(nn.Module):
    def __init__(self, input_dim, num_fields, num_row=2, num_heads=2, batch_norm=False, net_dropout=0.1):
        super(QuadraticLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim * num_row),
                                    nn.BatchNorm1d(input_dim * num_row) if batch_norm else nn.Identity(),
                                    nn.ReLU())
        if net_dropout > 0:
            self.dropout = nn.Dropout(net_dropout)
        self.net_dropout = net_dropout
        self.num_fields = num_fields
        self.num_row = num_row
        self.embedding_dim = input_dim // num_fields
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads

    def forward(self, x):  # Khatri-Rao product
        ego_x = x
        x = x.view(-1, self.num_fields, self.embedding_dim)
        multihead_x = torch.tensor_split(x, self.num_heads, dim=-1) # d = D/H
        multihead_x = torch.stack(multihead_x, dim=1).view(-1, self.num_heads, self.head_dim)  # B × H × d
        h = self.linear(ego_x).view(-1, self.num_heads, self.num_row, self.head_dim) # B × H × R × d
        if self.net_dropout > 0:
            h = self.dropout(h)
        x = torch.einsum("bhd,bhrd->bhrd", multihead_x, h).sum(dim=-2).view(-1, self.input_dim) + ego_x
        return x
