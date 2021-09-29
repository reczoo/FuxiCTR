# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingDictLayer, DNN_Layer, InnerProductLayer_v2, LR_Layer

class FLEN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FLEN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU", 
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(FLEN, self).__init__(feature_map, 
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.lr_layer = LR_Layer(feature_map, final_activation=None)
        self.mf_interaction = InnerProductLayer_v2(num_fields=3, output="element_wise")
        self.fm_interaction = InnerProductLayer_v2(output="bi_vector")
        self.dnn = DNN_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=None, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.r_ij = nn.Linear(3, 1, bias=False)
        self.r_mm = nn.Linear(3, 1, bias=False)
        self.w_FwBI = nn.Sequential(nn.Linear(embedding_dim + 1, embedding_dim + 1, bias=False),
                                    nn.ReLU())
        self.w_F = nn.Linear(dnn_hidden_units[-1] + embedding_dim + 1, 1, bias=False)
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb_dict = self.embedding_layer(X)
        emb_user = self.embedding_layer.dict2tensor(feature_emb_dict, feature_source="user")
        emb_item = self.embedding_layer.dict2tensor(feature_emb_dict, feature_source="item")
        emb_context = self.embedding_layer.dict2tensor(feature_emb_dict, feature_source="context")
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        lr_out = self.lr_layer(X)
        field_emb = torch.stack([emb_user.sum(dim=1), emb_item.sum(dim=1), emb_context.sum(dim=1)], dim=1)
        h_MF = self.r_ij(self.mf_interaction(field_emb).transpose(1, 2))
        h_FM = self.r_mm(torch.stack([self.fm_interaction(emb_user), self.fm_interaction(emb_item), 
                                      self.fm_interaction(emb_context)], dim=1).transpose(1, 2))
        h_FwBI = self.w_FwBI(torch.cat([lr_out, (h_MF + h_FM).squeeze(-1)], dim=-1))
        h_L = self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.w_F(torch.cat([h_FwBI, h_L], dim=-1))
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
