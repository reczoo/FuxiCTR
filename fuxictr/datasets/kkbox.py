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


import pandas as pd
from fuxictr.preprocess import FeatureProcessor as BaseFeatureProcessor

class FeatureProcessor(BaseFeatureProcessor):
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


