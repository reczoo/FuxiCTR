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

import random
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.python.keras.regularizers import l2, l1, l1_l2
from tensorflow.python.keras.initializers import *
import logging


def seed_everything(seed=2019):
    logging.info('Setting random seed={}'.format(seed))
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return tf.keras.layers.Activation("relu")
        elif activation.lower() == "sigmoid":
            return tf.keras.layers.Activation("sigmoid")
        elif activation.lower() == "tanh":
            return tf.keras.layers.Activation("tanh")
        elif activation.lower() == "softmax":
            return tf.keras.layers.Softmax()
        else:
            return getattr(tf.keras.layers, activation)()
    else:
        return activation

def get_optimizer(optimizer, learning_rate=1.0e-3):
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            return optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'ftrl':
            return optimizers.Ftrl(learning_rate=learning_rate, l1_regularization_strength=0.1)
        elif optimizer.lower() == 'adagrad':
            return optimizers.Adagrad(learning_rate=learning_rate)
        else:
            try:
                return getattr(optimizers, optimizer)(learning_rate=learning_rate)
            except:
                raise ValueError('optimizer={} is not supported.'.format(optimizer))
    return optimizer

def get_loss(loss):
    if isinstance(loss, str):
        if loss in ['bce', 'binary_crossentropy', 'binary_cross_entropy']:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            raise ValueError('loss={} is not supported.'.format(loss))
    return loss

def get_regularizer(reg):
    if type(reg) in [int, float]:
        return l2(reg)
    elif isinstance(reg, str):
        if '(' in reg:
            try:
                return eval(reg)
            except:
                raise ValueError('reg={} is not supported.'.format(reg))
    return reg

def get_initializer(initializer, seed=20222023):
    if isinstance(initializer, str):
        try:
            if '(' in initializer:
                return eval(initializer.rstrip(')') + ', seed={})'.format(seed))
            else:
                return eval(initializer)(seed=seed)
        except:
            raise ValueError("initializer={} not supported.".format(initializer))
    return initializer

