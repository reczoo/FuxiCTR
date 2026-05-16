# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from fuxictr.metrics import evaluate_metrics
from fuxictr.tensorflow.tf_utils import get_optimizer, get_loss
from fuxictr.utils import Monitor
import logging
from tqdm import tqdm


class BaseModel(Model):
    """Base class for ranking models in TensorFlow.

    Provides training loops, evaluation, checkpointing, early stopping, and
    metric computation for CTR prediction models.

    Args:
        feature_map (FeatureMap): Feature map object.
        model_id (str): Model identifier. Default: ``"BaseModel"``.
        task (str): Task type, e.g., ``binary_classification`` or ``regression``. Default: ``"binary_classification"``.
        monitor (str): Metric to monitor for early stopping. Default: ``"AUC"``.
        save_best_only (bool): Whether to save only the best checkpoint. Default: ``True``.
        monitor_mode (str): ``max`` or ``min`` for the monitored metric. Default: ``"max"``.
        early_stop_patience (int): Patience for early stopping. Default: ``2``.
        eval_steps (int): Evaluation frequency in steps; ``None`` means once per epoch. Default: ``None``.
        reduce_lr_on_plateau (bool): Whether to reduce learning rate on plateau. Default: ``True``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="BaseModel",
                 task="binary_classification",
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.valid_gen = None
        self._monitor_mode = monitor_mode
        self._monitor = Monitor(kv=monitor)
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._verbose = kwargs["verbose"]
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model.weights.h5"))
        self.validation_metrics = kwargs["metrics"]

    def compile(self, optimizer, loss, lr):
        """Configure the optimizer and loss function.

        Args:
            optimizer (str): Optimizer name.
            loss (str): Loss function name.
            lr (float): Learning rate.
        """
        self.optimizer = get_optimizer(optimizer, lr)
        self.loss_fn = get_loss(loss)

    def add_loss(self, inputs):
        """Compute the task loss without regularization.

        Args:
            inputs (dict): Batch data dictionary.

        Returns:
            tf.Tensor: Loss value.
        """
        return_dict = self(inputs, training=True)
        y_true = self.get_labels(inputs)
        loss = self.loss_fn(return_dict["y_pred"], y_true)
        return loss

    def compute_loss(self, inputs):
        """Compute the total loss including regularization.

        Args:
            inputs (dict): Batch data dictionary.

        Returns:
            tf.Tensor: Total loss value.
        """
        total_loss = self.add_loss(inputs) + sum(self.losses) # with regularization
        return total_loss

    def get_inputs(self, inputs, feature_source=None):
        """Extract input features from a batch dictionary.

        Args:
            inputs (dict): Batch data dictionary.
            feature_source (list, optional): Whitelist of feature sources to include.

        Returns:
            dict: Mapping of feature names to tensors.
        """
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            X_dict[feature] = inputs[feature]
        return X_dict

    def get_labels(self, inputs):
        """Extract labels from a batch dictionary.

        assert len(labels) == 1, "Please override get_labels() when using multiple labels!"

        Args:
            inputs (dict): Batch data dictionary.

        Returns:
            tf.Tensor: Label tensor.
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]]
        return y

    def get_group_id(self, inputs):
        """Extract group IDs from a batch dictionary.

        Args:
            inputs (dict): Batch data dictionary.

        Returns:
            tf.Tensor: Group ID tensor.
        """
        return inputs[self.feature_map.group_id]

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        """Reduce the learning rate by a multiplicative factor.

        Args:
            factor (float): Multiplicative factor applied to the current LR.
            min_lr (float): Lower bound for the learning rate.

        Returns:
            float: Updated learning rate.
        """
        self.optimizer.learning_rate = max(self.optimizer.learning_rate * factor, min_lr)
        return self.optimizer.lr.numpy()

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        """Train the model for a fixed number of epochs.

        Args:
            data_generator: Training data generator.
            epochs (int): Number of training epochs.
            validation_data: Validation data generator.
            max_gradient_norm (float): Maximum gradient norm for clipping.
            **kwargs: Additional keyword arguments.
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0

        # logging.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def train_epoch(self, data_generator):
        """Train the model for one epoch.

        Args:
            data_generator: Training data generator.
        """
        self._batch_index = 0
        train_loss = 0
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.numpy()
            if (self._eval_steps is not None) and (self._total_steps % self._eval_steps == 0):
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break
        if self._eval_steps is None:
            logging.info("Train loss: {:.6f}".format(train_loss / (self._batch_index + 1)))
            self.eval_step()

    @tf.function
    def train_step(self, batch_data):
        """Execute one training step on a single batch.

        Args:
            batch_data (dict): Batch data dictionary.

        Returns:
            tf.Tensor: Training loss value.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(batch_data)
            grads = tape.gradient(loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self._max_gradient_norm)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def eval_step(self):
        """Run a single evaluation step on the validation set."""
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        """Update checkpoints and determine whether to trigger early stopping.

        Args:
            logs (dict): Metric values from the latest evaluation.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch=={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def evaluate(self, data_generator, metrics=None):
        """Evaluate the model on a validation data generator.

        Args:
            data_generator: Validation data generator.
            metrics (list, optional): List of metric names to compute.

        Returns:
            dict: Mapping of metric names to computed values.
        """
        y_pred = []
        y_true = []
        group_id = []
        if self._verbose > 0:
            data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_data in data_generator:
            return_dict = self(batch_data, training=True)
            y_pred.extend(return_dict["y_pred"].numpy().reshape(-1))
            y_true.extend(self.get_labels(batch_data).numpy().reshape(-1))
            if self.feature_map.group_id is not None:
                group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None
        if metrics is not None:
            val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
        else:
            val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        """Compute evaluation metrics.

        Args:
            y_true (np.ndarray): Ground-truth labels.
            y_pred (np.ndarray): Predicted values.
            metrics (list): List of metric names.
            group_id (np.ndarray, optional): Group identifiers for grouped metrics.

        Returns:
            dict: Mapping of metric names to computed values.
        """
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def get_output_activation(self, task):
        """Get the output activation layer for a given task.

        Args:
            task (str): Task type, e.g., ``binary_classification`` or ``regression``.

        Returns:
            tf.keras.layers.Layer or callable: Output activation.

        Raises:
            NotImplementedError: If the task type is not supported.
        """
        if task == "binary_classification":
            return tf.keras.layers.Activation("sigmoid")
        elif task == "regression":
            return tf.identity
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

