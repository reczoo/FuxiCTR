# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import logging


def evaluate_metrics(y_true, y_pred, metrics):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            y_pred = np.argmax(y_pred, axis=1)
            result[metric] = accuracy_score(y_true, y_pred)
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result
