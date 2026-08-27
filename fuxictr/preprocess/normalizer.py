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

import numpy as np
import sklearn.preprocessing as skprocess


class Normalizer(object):
    """Normalize feature values using sklearn preprocessors or custom functions.

    Wraps sklearn scalers (``StandardScaler``, ``MinMaxScaler``) or any callable
    normalization function.

    Args:
        normalizer (str or callable): Name of sklearn scaler or a custom function.

    Raises:
        NotImplementedError: If ``normalizer`` is not a supported string.
    """

    def __init__(self, normalizer):
        if normalizer in ('StandardScaler', 'MinMaxScaler'):
            self.normalizer = normalizer
        elif callable(normalizer):
            # normalizer is a method
            self.normalizer = normalizer
        else:
            raise NotImplementedError('normalizer={}'.format(normalizer))

    def fit_from_stats(self, min_value=0, max_value=1, mean=None, std=None, count=None):
        """Fit ``StandardScaler`` / ``MinMaxScaler`` from aggregate statistics.

        Args:
            min_value (float, optional): Column min (MinMaxScaler).
            max_value (float, optional): Column max (MinMaxScaler).
            mean (float, optional): Column mean (StandardScaler).
            std (float, optional): Column std with ``ddof=0`` (StandardScaler;
                note polars ``std()`` defaults to ``ddof=1``).
            count (int, optional): Number of non-null values.
        """
        if self.normalizer == "StandardScaler":
            scaler = skprocess.StandardScaler()
            if std == 0: # constant column
                std = 1
            scaler.n_features_in_ = 1
            scaler.mean_ = np.array([mean])
            scaler.scale_ = np.array([std])
            scaler.var_ = np.array([std * std])
            scaler.n_samples_seen_ = count
            self.normalizer = scaler.transform
        elif self.normalizer == "MinMaxScaler":
            data_min = float(min_value) if min_value is not None else 0.0
            data_max = float(max_value) if max_value is not None else 1.0
            data_range = data_max - data_min
            if data_range == 0:
                data_range = 1.0
            scaler = skprocess.MinMaxScaler()
            scaler.n_features_in_ = 1
            scaler.data_min_ = np.array([data_min])
            scaler.data_max_ = np.array([data_max])
            scaler.data_range_ = np.array([data_range])
            scaler.scale_ = np.array([1.0 / data_range])
            scaler.min_ = np.array([-data_min / data_range])
            scaler.n_samples_seen_ = count
            self.normalizer = scaler.transform
        else:
            raise NotImplementedError("fit_from_stats only supports StandardScaler/MinMaxScaler")

    def transform(self, X):
        """Transform data using the fitted normalizer.

        Args:
            X (array-like): 1-D array of values to transform.

        Returns:
            numpy.ndarray: Normalized 1-D array.
        """
        return self.normalizer(X.reshape(-1, 1)).flatten()