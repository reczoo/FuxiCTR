# =========================================================================
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
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class SharedBottom(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="SharedBottom",
                 gpu=-1,
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 learning_rate=1e-3,
                 embedding_dim=10,
                 bottom_hidden_units=[64, 64, 64],
                 tower_hidden_units=[64, ],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SharedBottom, self).__init__(feature_map,
                                           task=task,
                                           loss_weight=loss_weight,
                                           num_tasks=num_tasks,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.bottom = MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                                hidden_units=bottom_hidden_units,
                                hidden_activations=hidden_activations,
                                output_activation=None,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm)
        self.tower = nn.ModuleList([MLP_Block(input_dim=bottom_hidden_units[-1],
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=hidden_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout,
                                              batch_norm=batch_norm)
                                    for _ in range(num_tasks)])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        bottom_output = self.bottom(feature_emb.flatten(start_dim=1)) # (?, bottom_hidden_units[-1])
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
