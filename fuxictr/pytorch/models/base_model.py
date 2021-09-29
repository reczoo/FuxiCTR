# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch.nn as nn
import numpy as np
import torch
import os
import logging
from ...metrics import evaluate_metrics
from ...pytorch.utils import set_device, set_optimizer, set_loss, set_regularizer
from ...utils import Monitor


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 patience=2, 
                 every_x_epochs=1, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = set_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._patience = patience
        self._every_x_epochs = every_x_epochs # float acceptable
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._feature_map = feature_map
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + "_model.ckpt"))
        self._validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]

    def compile(self, optimizer, loss, lr=1e-3):
        try:
            self.optimizer = set_optimizer(optimizer)(self.parameters(), lr=lr)
        except:
            raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
        try:
            self.loss_fn = getattr(torch.functional.F, set_loss(loss))
        except:
            try: 
                self.loss_fn = eval("losses." + loss)
            except:
                raise NotImplementedError("loss={} is not supported.".format(loss))

    def loss_with_reg(self, y_pred, y_true):
        total_loss = self.loss_fn(y_pred, y_true, reduction='mean')
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = set_regularizer(self._embedding_regularizer)
            net_reg = set_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                total_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                total_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return total_loss

    def init_weights(self, embedding_initializer=None):
        def _initialize(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        if "pretrained_emb" in self._feature_map.feature_specs[k]: # skip pretrained
                            continue
                        if embedding_initializer is not None:
                            try:
                                initializer = embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError("embedding_initializer={} is not supported."\
                                                          .format(embedding_initializer))
                        else:
                            nn.init.xavier_normal_(v.weight)
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(_initialize)
        
    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 1).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    def on_batch_end(self, batch, logs={}):
        self._total_batches += 1
        if (batch + 1) % self._every_x_batches == 0 or (batch + 1) % self._batches_per_epoch == 0:
            val_logs = self.evaluate_generator(self.valid_gen)
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info("--- {}/{} batches finished ---".format(batch + 1, self._batches_per_epoch))

    def reduce_learning_rate(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({}) STOP: {:.6f} !".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.reduce_learning_rate()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({}): {:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps * self._every_x_epochs >= self._patience:
            self._stop_training = True
            logging.info("Early stopping at epoch={:g}".format(epoch))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)
            
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
        self.to(device=self.device)
        
        logging.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_on_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def train_on_epoch(self, data_generator, epoch):
        epoch_loss = 0
        model = self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            return_dict = model.forward(batch_data)
            loss = return_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.on_batch_end(batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(batch_data[1].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_logs = evaluate_metrics(y_true, y_pred, self._validation_metrics)
            return val_logs

    def to_device(self):
        self.to(device=self.device)
                
    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint, map_location=self.device))

    def get_final_activation(self, task="binary_classification"):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "multi_classification":
            return nn.Softmax(dim=-1)
        elif task == "regression":
            return None
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

