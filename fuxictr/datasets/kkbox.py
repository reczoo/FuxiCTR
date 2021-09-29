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
    def extract_country_code(self, df, col_name):
        return df[col_name].apply(lambda isrc: isrc[0:2] if not pd.isnull(isrc) else "")

    def bucketize_age(self, df, col_name):
        def _bucketize(age):
            if pd.isnull(age):
                return ""
            else:
                age = float(age)
                if age < 1 or age > 95:
                    return ""
                elif age <= 10:
                    return "1"
                elif age <=20:
                    return "2"
                elif age <=30:
                    return "3"
                elif age <=40:
                    return "4"
                elif age <=50:
                    return "5"
                elif age <=60:
                    return "6"
                else:
                    return "7"
        return df[col_name].apply(_bucketize)


