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
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, MultiHeadTargetAttention
from fuxictr.utils import not_in_whitelist


class SIM(BaseModel):
    """Search-based Interest Model (SIM) with two-stage attention.

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs.
        model_id (str): Model identifier string. Default: ``"SIM"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        dnn_hidden_units (list): Hidden units of the DNN. Default: ``[512, 128, 64]``.
        dnn_activations (str): Activation function for DNN layers. Default: ``"ReLU"``.
        attention_dropout (float): Dropout rate for attention layers. Default: ``0``.
        attention_dim (int): Dimension of the attention space. Default: ``64``.
        num_heads (int): Number of attention heads. Default: ``1``.
        gsu_type (str): Type of general search unit, currently only ``"soft"`` is supported.
            Default: ``"soft"``.
        short_seq_len (int): Length of the short sequence for attention. Default: ``50``.
        topk (int): Number of top-k items to retrieve. Default: ``50``.
        alpha (float): Weight for the auxiliary GSU loss. Default: ``1``.
        beta (float): Weight for the main ESU loss. Default: ``1``.
        learning_rate (float): Learning rate for training. Default: ``1e-3``.
        embedding_dim (int): Embedding dimension of features. Default: ``10``.
        net_dropout (float): Dropout rate for DNN. Default: ``0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``False``.
        accumulation_steps (int): Gradient accumulation steps. Default: ``1``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network weights. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="SIM",
                 gpu=-1,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dropout=0,
                 attention_dim=64,
                 num_heads=1,
                 gsu_type="soft",
                 short_seq_len=50,
                 topk=50,
                 alpha=1,
                 beta=1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SIM, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.topk = topk
        self.short_seq_len = short_seq_len
        self.alpha = alpha
        self.beta = beta
        assert gsu_type == "soft", "Only support soft search currently!"
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.W_a = nn.Linear(self.item_info_dim, attention_dim, bias=False)
        self.W_b = nn.Linear(self.item_info_dim, attention_dim, bias=False)
        self.short_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout)
        self.long_attention = MultiHeadTargetAttention(self.item_info_dim,
                                                       attention_dim,
                                                       num_heads,
                                                       attention_dropout)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.dnn_aux = MLP_Block(input_dim=input_dim,
                                 output_dim=1,
                                 hidden_units=dnn_hidden_units,
                                 hidden_activations=dnn_activations,
                                 output_activation=self.output_activation, 
                                 dropout_rates=net_dropout,
                                 batch_norm=batch_norm)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim * 2
        self.dnn = MLP_Block(input_dim=input_dim,
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
        """Forward pass of SIM.

        Args:
            inputs: Model inputs.

        Returns:
            dict: Dictionary containing ``y_pred`` and ``y_aux``.
        """
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            emb_out = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(emb_out)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]

        # short interest attention
        short_seq_emb = item_feat_emb[:, -self.short_seq_len:-1, :]
        short_mask = mask[:, -self.short_seq_len:-1]
        short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)

        # first stage
        long_seq_emb = item_feat_emb[:, 0:-1, :]
        q = self.W_a(target_emb).unsqueeze(1)
        k = self.W_b(long_seq_emb)
        qk = torch.bmm(q, k.transpose(-1, -2)).squeeze(1) * mask
        pooled_u_rep = torch.bmm(qk.unsqueeze(1), long_seq_emb).squeeze(1)
        emb_list += [target_emb, pooled_u_rep]
        y_aux = self.dnn_aux(torch.cat(emb_list, dim=-1))
        topk = min(self.topk, qk.shape[1]) # make sure input seq_len >= topk
        topk_index = qk.topk(topk, dim=1, largest=True, sorted=True)[1]
        topk_emb = torch.gather(long_seq_emb, 1,
                                topk_index.unsqueeze(-1).expand(-1, -1, long_seq_emb.shape[-1]))
        topk_mask = torch.gather(mask, 1, topk_index)

        # second stage
        long_interest_emb = self.long_attention(target_emb, topk_emb, topk_mask)
        emb_list = emb_list[0:-1] + [short_interest_emb, long_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred, "y_aux": y_aux}
        return return_dict

    def add_loss(self, return_dict, y_true):
        """Compute combined GSU and ESU loss.

        Args:
            return_dict: Dictionary with model outputs.
            y_true: Ground truth labels.

        Returns:
            Tensor: Combined loss value.
        """
        loss_gsu = self.loss_fn(return_dict["y_aux"], y_true, reduction='mean')
        loss_esu = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return self.alpha * loss_gsu + self.beta * loss_esu

    def get_inputs(self, inputs, feature_source=None):
        """Extract input tensors from the data batch.

        Args:
            inputs: Raw input batch.
            feature_source: Optional feature source filter.

        Returns:
            tuple: ``(X_dict, item_dict, mask)`` tensors moved to the model device.
        """
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
        """Get group IDs from the input batch.

        Args:
            inputs: Input batch.

        Returns:
            Tensor: Group ID tensor.
        """
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        """Perform a single training step.

        Args:
            batch_data: A batch of training data.

        Returns:
            Tensor: The computed loss value.
        """
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
