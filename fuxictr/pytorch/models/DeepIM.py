# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from .base_model import BaseModel
from ..layers import EmbeddingLayer, MLP_Layer

class DeepIM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepIM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 im_order=2, 
                 im_batch_norm=False,
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 net_batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DeepIM, self).__init__(feature_map, 
                                     model_id=model_id,
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.im_layer = InteractionMachine(embedding_dim, im_order, im_batch_norm)
        self.dnn = MLP_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             final_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=net_batch_norm) \
                   if hidden_units is not None else None
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.apply(self.init_weights)
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.im_layer(feature_emb)
        if self.dnn is not None:
            y_pred += self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.final_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class InteractionMachine(nn.Module):
    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, "order={} is not supported.".format(order)
        self.order = order
        self.bn = nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = nn.Linear(order * embedding_dim, 1)
        
    def second_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2)
                + 8 * p1 * p3 - 6 * p4) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4
                - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120

    def forward(self, X):
        out = []
        Q = X
        if self.order >= 1:
            p1 = Q.sum(dim=1)
            out.append(p1)
            if self.order >= 2:
                Q = Q * X
                p2 = Q.sum(dim=1)
                out.append(self.second_order(p1, p2))
                if self.order >= 3:
                    Q = Q * X
                    p3 = Q.sum(dim=1)
                    out.append(self.third_order(p1, p2, p3))
                    if self.order >= 4:
                        Q = Q * X
                        p4 = Q.sum(dim=1)
                        out.append(self.fourth_order(p1, p2, p3, p4))
                        if self.order == 5:
                            Q = Q * X
                            p5 = Q.sum(dim=1)
                            out.append(self.fifth_order(p1, p2, p3, p4, p5))
        out = torch.cat(out, dim=-1)
        if self.bn is not None:
            out = self.bn(out)
        y = self.fc(out)
        return y
