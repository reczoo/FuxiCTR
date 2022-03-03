# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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


import pandas as pd
import numpy as np
import os
from ..features import FeatureEncoder as BaseFeatureEncoder
from datetime import datetime, date


class FeatureEncoder(BaseFeatureEncoder):
    def convert_hour(self, df, col_name):
        return df['time_stamp'].apply(lambda ts: ts[11:13])

    def convert_weekday(self, df, col_name):
        def _convert_weekday(timestamp):
            dt = date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))
            return dt.strftime('%w')
        return df['time_stamp'].apply(_convert_weekday)

    def convert_weekend(self, df, col_name):
        def _convert_weekend(timestamp):
            dt = date(int(timestamp[0:4]), int(timestamp[5:7]), int(timestamp[8:10]))
            return '1' if dt.strftime('%w') in ['6', '0'] else '0'
        return df['time_stamp'].apply(_convert_weekend)