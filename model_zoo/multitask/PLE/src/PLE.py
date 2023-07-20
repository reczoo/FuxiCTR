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
from fuxictr.pytorch.torch_utils import get_activation


class CGC_Layer(nn.Module):
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(CGC_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts 
        self.num_specific_experts = num_specific_experts 
        self.num_tasks = num_tasks 
        self.shared_experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_specific_experts)]) for _ in range(num_tasks)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             output_dim=num_specific_experts+num_shared_experts if i < num_tasks else num_shared_experts,
                                             hidden_units=gate_hidden_units,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for i in range(self.num_tasks+1)])
        self.gate_activation = get_activation('softmax')
    def forward(self, x, require_gate=False):
        """
        x: list, len(x)==num_tasks+1
        """
        specific_expert_outputs = [] 
        shared_expert_outputs = []
        # specific experts
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
        # shared experts 
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        # gate 
        cgc_outputs = [] 
        gates = [] 
        for i in range(self.num_tasks+1):
            if i < self.num_tasks:
                # for specific experts
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1) # (?, num_specific_experts+num_shared_experts, dim)
                gate = self.gate_activation(self.gate[i](x[i])) # (?, num_specific_experts+num_shared_experts)
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1) # (?, dim)
                cgc_outputs.append(cgc_output)
            else: 
                # for shared experts 
                gate_input = torch.stack(shared_expert_outputs, dim=1) # (?, num_shared_experts, dim)
                gate = self.gate_activation(self.gate[i](x[-1])) # (?, num_shared_experts)
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1) # (?, dim)
                cgc_outputs.append(cgc_output)
        if require_gate:
            return cgc_outputs, gates 
        else: 
            return cgc_outputs

class PLE(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=1,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(PLE, self).__init__(feature_map,
                                   task=task,
                                   num_tasks=num_tasks,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_layers = num_layers
        self.cgc_layers = nn.ModuleList([CGC_Layer(num_shared_experts,
                                                   num_specific_experts,
                                                   num_tasks,
                                                   input_dim= embedding_dim * feature_map.num_fields if i==0 else expert_hidden_units[-1],
                                                   expert_hidden_units= expert_hidden_units,
                                                   gate_hidden_units=gate_hidden_units,
                                                   hidden_activations=hidden_activations,
                                                   net_dropout=net_dropout,
                                                   batch_norm=batch_norm) for i in range(self.num_layers)])
        self.tower = nn.ModuleList([MLP_Block(input_dim=expert_hidden_units[-1],
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
        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks+1)]
        for i in range(self.num_layers):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs
        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
