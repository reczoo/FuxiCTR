# =========================================================================
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
from fuxictr.preprocess import FeatureProcessor as BaseFeatureProcessor


class FeatureProcessor(BaseFeatureProcessor):
    def convert_to_bucket(self, df, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        return df[col_name].map(_convert_to_bucket).astype(int)


	


