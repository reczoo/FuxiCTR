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
import polars as pl


class CustomizedFeatureProcessor(FeatureProcessor):
    """KKBox dataset feature processor with ISRC and age helpers."""

    def extract_country_code(self, col_name):
        """Extract the 2-letter country code from an ISRC string.

        Args:
            col_name (str): Name of the ISRC column.

        Returns:
            pl.Expr: Polars expression yielding the country code or an empty string.
        """
        return pl.col(col_name).str.slice(0, 2).fill_null("")

    def bucketize_age(self, col_name):
        """Bucketize a raw age value into one of seven string buckets.

        Args:
            col_name (str): Name of the age column.

        Returns:
            pl.Expr: Polars expression yielding the bucket id as a string.
        """
        age = pl.col(col_name).cast(pl.Float64)
        in_range = age.is_between(1, 95, closed="both")
        return (
            pl.when(age.is_null() | ~in_range).then(pl.lit(""))
            .when(age <= 10).then(pl.lit("1"))
            .when(age <= 20).then(pl.lit("2"))
            .when(age <= 30).then(pl.lit("3"))
            .when(age <= 40).then(pl.lit("4"))
            .when(age <= 50).then(pl.lit("5"))
            .when(age <= 60).then(pl.lit("6"))
            .otherwise(pl.lit("7"))
        )
