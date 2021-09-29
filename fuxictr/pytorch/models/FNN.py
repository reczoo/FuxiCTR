# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
import numpy as np
import logging
import shutil
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, DNN_Layer, InnerProductLayer_v2

class FNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FNN", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False,
                 fm_embedding_regularizer=None,
                 dnn_embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(FNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=dnn_embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.dnn_embedding_regularizer = dnn_embedding_regularizer
        self.fm_embedding_regularizer = fm_embedding_regularizer
        # A trick for quick one-hot encoding in LR
        self.lr_embedding_layer = EmbeddingLayer_v3(feature_map, 1)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.fm_embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        self.inner_product_layer = InnerProductLayer_v2(output="sum")
        self.dnn = DNN_Layer(input_dim=(embedding_dim + 1) * feature_map.num_fields + 1,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)
        self.final_activation = self.get_final_activation(task)
        self.learning_rate = learning_rate
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        lr_weights = self.lr_embedding_layer(X)
        feature_emb = self.fm_embedding_layer(X)
        if self._pretrain: # FM branch
            lr_out = lr_weights.sum(dim=1) + self.bias
            fm_out = self.inner_product_layer(feature_emb)
            y_pred = lr_out + fm_out
            self._embedding_regularizer = self.fm_embedding_regularizer
        else: # DNN branch
            flat_emb = torch.cat([lr_weights, feature_emb], dim=-1).flatten(start_dim=1)
            y_pred = self.dnn(torch.cat([self.bias.repeat(flat_emb.size(0), 1), flat_emb], dim=-1))
            self._embedding_regularizer = self.dnn_embedding_regularizer
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def fit_generator(self, data_generator, epochs=1, validation_data=None,
                      verbose=0, max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(data_generator)
        self._every_x_batches = int(np.ceil(self._every_x_epochs * self._batches_per_epoch))
        self._stop_training = False
        self._verbose = verbose
        self._pretrain = True
        self.to(device=self.device)
        
        logging.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        k = 0
        while k < 2:
            k += 1
            logging.info("************ Epoch=1 start ************")
            for epoch in range(epochs):
                epoch_loss = self.train_on_epoch(data_generator, epoch)
                logging.info("Train loss: {:.6f}".format(epoch_loss))
                logging.info("************ Epoch={} end ************".format(epoch + 1))
                if self._stop_training:
                    self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
                    self._stopping_steps = 0
                    self._total_batches = 0
                    self._stop_training = False
                    self._pretrain = False
                    self.load_weights(self.checkpoint) # load best weights for fine tuning
                    shutil.copyfile(self.checkpoint, self.checkpoint + "_fm.pretrain")
                    self.reduce_learning_rate(min_lr=self.learning_rate) # set learning rate for DNN
                    break
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

