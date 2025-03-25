# =========================================================================
# Copyright (C) 2025. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. TransAct Authors from Pinterest. Modified from their original implementation.
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


""" This model implements the paper: Xia et al., TransAct: Transformer-based Realtime User Action
    Model for Recommendation at Pinterest, KDD 2023.
    [PDF] https://arxiv.org/abs/2306.00248
    [Code] https://github.com/pinterest/transformer_user_action
"""


import torch
from torch import nn
import numpy as np
import random
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2
from fuxictr.utils import not_in_whitelist


class TransAct(BaseModel):
    """
    The TransAct model class that implements transformer-based realtime user action model.
    Make sure the behavior sequences are sorted in chronological order and padded in the left part.

    Args:
        feature_map: A FeatureMap instance used to store feature specs (e.g., vocab_size).
        model_id: Equivalent to model class name by default, which is used in config to determine 
            which model to call.
        gpu: gpu device used to load model.
        hidden_activations: hidden activations used in MLP blocks (default="ReLU").
        dcn_cross_layers: number of cross layers in DCNv2 (default=3).
        dcn_hidden_units: hidden units of deep part in DCNv2 (default=[256, 128, 64]).
        mlp_hidden_units: hidden units of MLP on top of DCNv2 (default=[]).
        num_heads: number of heads of transformer (default=1).
        transformer_layers: number of stacked transformer layers used in TransAct (default=1).
        transformer_dropout: dropout rate used in transformer (default=0).
        dim_feedforward: FFN dimension in transformer (default=512)
        learning_rate: learning rate for training (default=1e-3).
        embedding_dim: embedding dimension of features (default=64).
        net_dropout: dropout rate for deep part in DCNv2 (default=0).
        batch_norm: whether to apply batch normalization in DCNv2 (default=False).
        target_item_field (List[tuple] or List[str]): which field is used for target item
            embedding. When tuple is applied, the fields in each tuple are concatenated, e.g.,
            item_id and cate_id can be concatenated as target item embedding.
        sequence_item_field (List[tuple] or List[str]): which field is used for sequence item
            embedding. When tuple is applied, the fields in each tuple are concatenated.
        first_k_cols: number of hidden representations to pick as transformer output (default=1).
        use_time_window_mask (Boolean): whether to use time window mask in TransAct (default=False).
        time_window_ms: time window in ms to mask the most recent behaviors (default=86400000).
        concat_max_pool (Boolean): whether cancate max pooling result in transformer output
            (default=True).
        embedding_regularizer: regularization term used for embedding parameters (default=0).
        net_regularizer: regularization term used for network parameters (default=0).
    """
    def __init__(self,
                 feature_map,
                 model_id="TransAct",
                 gpu=-1,
                 hidden_activations="ReLU",
                 dcn_cross_layers=3,
                 dcn_hidden_units=[256, 128, 64],
                 mlp_hidden_units=[],
                 num_heads=1,
                 transformer_layers=1,
                 transformer_dropout=0,
                 dim_feedforward=512,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 net_dropout=0,
                 batch_norm=False,
                 first_k_cols=1,
                 use_time_window_mask=False,
                 time_window_ms=86400000,
                 concat_max_pool=True,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super().__init__(feature_map, 
                         model_id=model_id,
                         gpu=gpu, 
                         embedding_regularizer=embedding_regularizer, 
                         net_regularizer=net_regularizer,
                         **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        transformer_in_dim = self.item_info_dim * 2
        seq_out_dim = (first_k_cols + int(concat_max_pool)) * transformer_in_dim
        self.transformer_encoders = (
            TransActTransformer(transformer_in_dim,
                                dim_feedforward=dim_feedforward,
                                num_heads=num_heads,
                                dropout=transformer_dropout,
                                transformer_layers=transformer_layers,
                                use_time_window_mask=use_time_window_mask,
                                time_window_ms=time_window_ms,
                                first_k_cols=first_k_cols,
                                concat_max_pool=concat_max_pool)
        )
        dcn_in_dim = feature_map.sum_emb_out_dim() + seq_out_dim
        self.crossnet = CrossNetV2(dcn_in_dim, dcn_cross_layers)
        self.parallel_dnn = MLP_Block(input_dim=dcn_in_dim,
                                      output_dim=None, # output hidden layer
                                      hidden_units=dcn_hidden_units,
                                      hidden_activations=hidden_activations,
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)
        dcn_out_dim = dcn_in_dim + dcn_hidden_units[-1]
        self.mlp = MLP_Block(input_dim=dcn_out_dim,
                             output_dim=1,
                             hidden_units=mlp_hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, pad_mask = self.get_inputs(inputs)
        feature_emb = []
        if batch_dict: # not empty
            emb_out = self.embedding_layer(batch_dict, flatten_emb=True)
            feature_emb.append(emb_out)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = pad_mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        transformer_out = self.transformer_encoders(
            target_emb, sequence_emb, mask=~pad_mask.bool()
        )
        feature_emb += [target_emb, transformer_out]
        dcn_in_emb = torch.cat(feature_emb, dim=-1)
        cross_out = self.crossnet(dcn_in_emb)
        dnn_out = self.parallel_dnn(dcn_in_emb)
        y_pred = self.mlp(torch.cat([cross_out, dnn_out], dim=-1))
        return_dict = {"y_pred": y_pred}
        return return_dict

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


class TransActTransformer(nn.Module):
    def __init__(self,
                 transformer_in_dim,
                 dim_feedforward=64,
                 num_heads=1,
                 dropout=0,
                 transformer_layers=1,
                 use_time_window_mask=False,
                 time_window_ms=86400000, # recent 24h
                 first_k_cols=1,
                 concat_max_pool=True):
        super(TransActTransformer, self).__init__()
        self.use_time_window_mask = use_time_window_mask
        self.time_window_ms = time_window_ms
        self.concat_max_pool = concat_max_pool
        self.first_k_cols = first_k_cols
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        if self.concat_max_pool:
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def forward(self, target_emb, sequence_emb, time_interval_seq=None, mask=None):
        # concat action sequence emb with target emb
        seq_len = sequence_emb.size(1)
        concat_seq_emb = torch.cat([sequence_emb,
                                    target_emb.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        # get sequence mask (1's are masked)
        key_padding_mask = self.adjust_mask(mask) # keep the last dim
        if self.use_time_window_mask and self.training:
            rand_time_window_ms = random.randint(0, self.time_window_ms)
            time_window_mask = (time_interval_seq < rand_time_window_ms)
            key_padding_mask = torch.bitwise_or(key_padding_mask, time_window_mask)
        tfmr_out = self.transformer_encoder(src=concat_seq_emb,
                                            src_key_padding_mask=key_padding_mask)
        tfmr_out = tfmr_out.masked_fill(
            key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), 0.
        )
        # process the transformer output
        output_concat = []
        output_concat.append(tfmr_out[:, -self.first_k_cols:].flatten(start_dim=1))
        if self.concat_max_pool:
            # Apply max pooling to the transformer output
            tfmr_out = tfmr_out.masked_fill(
                key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), -1e9
            )
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        return torch.cat(output_concat, dim=-1)

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        fully_masked = mask.all(dim=-1)
        mask[fully_masked, -1] = 0
        return mask
