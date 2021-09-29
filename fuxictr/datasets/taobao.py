# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

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