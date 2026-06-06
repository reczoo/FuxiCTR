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


from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict


def evaluate_metrics(y_true, y_pred, metrics, group_id=None):
    """Evaluate a list of metrics on predictions.

    Supports ``logloss``, ``AUC``, ``gAUC``, ``avgAUC``, ``MRR``, and ``NDCG@k``.
    Group-level metrics (``gAUC``, ``avgAUC``, ``MRR``, ``NDCG``) require
    ``group_id`` to be provided.

    Args:
        y_true (array-like): Ground-truth binary labels.
        y_pred (array-like): Predicted probabilities or scores.
        metrics (list): List of metric names to compute.
        group_id (array-like, optional): Group identifiers for group-level metrics.

    Returns:
        OrderedDict: Mapping from metric name to computed value.

    Raises:
        ValueError: If an unsupported metric is requested.
    """
    return_dict = OrderedDict()
    group_metrics = []
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            return_dict[metric] = log_loss(y_true, y_pred)
        elif metric == 'AUC':
            return_dict[metric] = roc_auc_score(y_true, y_pred)
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            return_dict[metric] = 0
            group_metrics.append(metric)
        else:
            raise ValueError("metric={} not supported.".format(metric))
    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = []
        for metric in group_metrics:
            try:
                metric_funcs.append(eval(metric))
            except:
                raise NotImplementedError('metrics={} not implemented.'.format(metric))
        score_df = pd.DataFrame({"group_index": group_id,
                                 "y_true": y_true,
                                 "y_pred": y_pred})
        results = []
        pool = mp.Pool(processes=mp.cpu_count() // 2)
        for idx, df in score_df.groupby("group_index"):
            results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        sum_results = np.array(results).sum(0)
        average_result = list(sum_results[:, 0] / sum_results[:, 1])
        return_dict.update(dict(zip(group_metrics, average_result)))
    return return_dict

def evaluate_block(df, metric_funcs):
    """Evaluate a list of metric functions on a single group DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with ``y_true`` and ``y_pred`` columns.
        metric_funcs (list): List of callable metric functions.

    Returns:
        list: List of ``(value, weight)`` tuples.
    """
    res_list = []
    for fn in metric_funcs:
        v = fn(df.y_true.values, df.y_pred.values)
        if type(v) == tuple:
            res_list.append(v)
        else: # add group weight
            res_list.append((v, 1))
    return res_list

def avgAUC(y_true, y_pred):
    """Compute average AUC used in MIND news recommendation.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted scores.

    Returns:
        tuple: ``(auc_value, weight)`` or ``(0, 0)`` for all-same-label groups.
    """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        return (auc, 1)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def gAUC(y_true, y_pred):
    """Compute group AUC defined in the DIN paper.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted scores.

    Returns:
        tuple: ``(weighted_auc, n_samples)`` or ``(0, 0)`` for all-same-label groups.
    """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        n_samples = len(y_true)
        return (auc * n_samples, n_samples)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def MRR(y_true, y_pred):
    """Compute Mean Reciprocal Rank.

    Args:
        y_true (array-like): Ground-truth binary relevance labels.
        y_pred (array-like): Predicted scores for ranking.

    Returns:
        float: MRR score.
    """
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / (np.sum(y_true) + 1e-12)
    return mrr


class NDCG(object):
    """Normalized discounted cumulative gain metric.

    Computes DCG at a given cutoff ``k`` and normalizes by the ideal DCG.

    Args:
        k (int): Rank cutoff for DCG computation. Default: ``1``.
    """

    def __init__(self, k=1):
        self.topk = k

    def dcg_score(self, y_true, y_pred):
        """Compute discounted cumulative gain at ``self.topk``.

        Args:
            y_true (array-like): Ground-truth relevance labels.
            y_pred (array-like): Predicted scores for ranking.

        Returns:
            float: DCG score.
        """
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:self.topk])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        """Compute NDCG at ``self.topk``.

        Args:
            y_true (array-like): Ground-truth relevance labels.
            y_pred (array-like): Predicted scores for ranking.

        Returns:
            float: NDCG score in ``[0, 1]``.
        """
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return dcg / (idcg + 1e-12)


