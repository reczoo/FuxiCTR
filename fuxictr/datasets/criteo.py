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
    def convert_to_bucket(self, df, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        return df[col_name].map(_convert_to_bucket).astype(int)
