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


import sys
import os
import numpy as np
import torch
from torch import nn
import random
from functools import partial
import re


def seed_everything(seed=1029):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value. Default: ``1029``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(gpu=-1):
    """Get a PyTorch compute device.

    Args:
        gpu (int): GPU device index. If negative or CUDA unavailable, returns CPU.
            Default: ``-1``.

    Returns:
        torch.device: The selected compute device.
    """
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device

def get_optimizer(optimizer, params, lr):
    """Get a PyTorch optimizer instance.

    Args:
        optimizer (str or torch.optim.Optimizer): Optimizer name or class.
        params (iterable): Model parameters to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        NotImplementedError: If the optimizer name is not recognized.
    """
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer

def get_loss(loss):
    """Get a PyTorch loss function.

    Args:
        loss (str or callable): Loss function name (e.g., ``"binary_cross_entropy"``)
            or a callable loss function.

    Returns:
        callable: Loss function.

    Raises:
        NotImplementedError: If the loss name is not supported.
    """
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try:
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))
    return loss_fn

def get_regularizer(reg):
    """Parse a regularization specification into (p_norm, weight) tuples.

    Supports ``float`` (L2), ``"l1(x)"``, ``"l2(x)"``, and ``"l1_l2(x,y)"`` strings.

    Args:
        reg (float or str): Regularization specification.

    Returns:
        list: List of ``(p_norm, weight)`` tuples.

    Raises:
        NotImplementedError: If the regularization format is not supported.
    """
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def get_activation(activation, hidden_units=None):
    """Get a PyTorch activation module or function.

    Args:
        activation (str, list, or callable): Activation name (e.g., ``"relu"``,
            ``"dice"``, ``"prelu"``), a list of names, or a callable.
        hidden_units (int or list, optional): Number of hidden units. Required
            for ``"prelu"`` and ``"dice"``.

    Returns:
        torch.nn.Module or callable: Activation module/function, or a list thereof.
    """
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == "dice":
            from fuxictr.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    """Get a PyTorch weight initializer.

    Args:
        initializer (str or callable): Initializer name (e.g., ``"torch.nn.init.xavier_uniform_"``)
            or a callable initializer function.

    Returns:
        callable: Initializer function.

    Raises:
        ValueError: If the initializer name is not supported.
    """
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer
