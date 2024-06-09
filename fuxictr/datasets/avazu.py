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


from fuxictr.preprocess import FeatureProcessor
from datetime import datetime, date
import polars as pl


class CustomizedFeatureProcessor(FeatureProcessor):
    def convert_weekday(self, col_name=None):
        def _convert_weekday(timestamp):
            dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
            return int(dt.strftime('%w'))
        return pl.col("hour").apply(_convert_weekday)

    def convert_weekend(self, col_name=None):
        def _convert_weekend(timestamp):
            dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
            return 1 if dt.strftime('%w') in ['6', '0'] else 0
        return pl.col("hour").apply(_convert_weekend)

    def convert_hour(self, col_name=None):
        return pl.col("hour").apply(lambda x: int(x[6:8]))
