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
import sklearn.preprocessing as sklearn_preprocess


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
        if not callable(normalizer):
            self.callable = False
            if normalizer in ['StandardScaler', 'MinMaxScaler']:
                self.normalizer = getattr(sklearn_preprocess, normalizer)()
            else:
                raise NotImplementedError('normalizer={}'.format(normalizer))
        else:
            # normalizer is a method
            self.normalizer = normalizer
            self.callable = True

    def fit(self, X):
        """Fit the normalizer on data.

        Args:
            X (array-like): 1-D array of values to fit.
        """
        if not self.callable:
            self.normalizer.fit(X.reshape(-1, 1))

    def transform(self, X):
        """Transform data using the fitted normalizer.

        Args:
            X (array-like): 1-D array of values to transform.

        Returns:
            numpy.ndarray: Normalized 1-D array.
        """
        if self.callable:
            return self.normalizer(X)
        else:
            return self.normalizer.transform(X.reshape(-1, 1)).flatten()
