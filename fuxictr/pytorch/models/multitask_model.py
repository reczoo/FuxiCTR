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
import numpy as np
import torch
import os, sys
import logging
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from tqdm import tqdm
from collections import defaultdict


class MultiTaskModel(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MultiTaskModel",
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(MultiTaskModel, self).__init__(feature_map=feature_map,
                                             model_id=model_id,
                                             task="binary_classification",
                                             gpu=gpu,
                                             loss_weight=loss_weight,
                                             monitor=monitor,
                                             save_best_only=save_best_only,
                                             monitor_mode=monitor_mode,
                                             early_stop_patience=early_stop_patience,
                                             eval_steps=eval_steps,
                                             embedding_regularizer=embedding_regularizer,
                                             net_regularizer=net_regularizer,
                                             reduce_lr_on_plateau=reduce_lr_on_plateau,
                                             **kwargs)
        self.device = get_device(gpu)
        self.num_tasks = num_tasks
        self.loss_weight = loss_weight
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        if isinstance(loss, list):
            self.loss_fn = [get_loss(l) for l in loss]
        else:
            self.loss_fn = [get_loss(loss) for _ in range(self.num_tasks)]

    def get_labels(self, inputs):
        """ Override get_labels() to use multiple labels """
        labels = self.feature_map.labels
        y = [inputs[labels[i]].to(self.device).float().view(-1, 1)
             for i in range(len(labels))]
        return y

    def regularization_loss(self):
        reg_loss = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_loss

    def add_loss(self, return_dict, y_true):
        labels = self.feature_map.labels
        loss = [self.loss_fn[i](return_dict["{}_pred".format(labels[i])], y_true[i], reduction='mean')
                for i in range(len(labels))]
        if self.loss_weight == 'EQ':
            # Default: All losses are weighted equally
            loss = torch.sum(torch.stack(loss))
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss
    
    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            y_true_all = defaultdict(list)
            labels = self.feature_map.labels
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                batch_y_true = self.get_labels(batch_data)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["{}_pred".format(labels[i])].data.cpu().numpy().reshape(-1))
                    y_true_all[labels[i]].extend(batch_y_true[i].data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            all_val_logs = {}
            mean_val_logs = defaultdict(list)
            group_id = np.array(group_id) if len(group_id) > 0 else None

            for i in range(len(labels)):
                y_pred = np.array(y_pred_all[labels[i]], np.float64)
                y_true = np.array(y_true_all[labels[i]], np.float64)
                if metrics is not None:
                    val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                else:
                    val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                logging.info('[Task: {}][Metrics] '.format(labels[i]) + ' - '.join(
                    '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                for k, v in val_logs.items():
                    all_val_logs['{}_{}'.format(labels[i], k)] = v
                    mean_val_logs[k].append(v)
            for k, v in mean_val_logs.items():
                mean_val_logs[k] = np.mean(v)
            all_val_logs.update(mean_val_logs)
            return all_val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            labels = self.feature_map.labels
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["{}_pred".format(labels[i])].data.cpu().numpy().reshape(-1))
        return y_pred_all
