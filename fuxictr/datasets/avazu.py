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
    """Avazu dataset feature processor with timestamp-derived features."""

    def convert_weekday(self, col_name=None):
        """Extract the weekday (0-6) from the ``hour`` timestamp column.

        Args:
            col_name (str, optional): Unused; kept for API compatibility.

        Returns:
            pl.Expr: Polars expression that yields the weekday as an integer.
        """
        return (
            pl.col("hour")
            .str.slice(0, 6)
            .str.to_date(format="%y%m%d")
            .dt.weekday()
            % 7
        ).cast(pl.Int32)

    def convert_weekend(self, col_name=None):
        """Extract a weekend indicator (1 if Sat/Sun, else 0) from ``hour``.

        Args:
            col_name (str, optional): Unused; kept for API compatibility.

        Returns:
            pl.Expr: Polars expression that yields 1 for weekend days, 0 otherwise.
        """
        return (
            pl.col("hour")
            .str.slice(0, 6)
            .str.to_date(format="%y%m%d")
            .dt.weekday()
            % 7
        ).is_in([0, 6]).cast(pl.Int32)

    def convert_hour(self, col_name=None):
        """Extract the hour-of-day (0-23) from the ``hour`` timestamp column.

        Args:
            col_name (str, optional): Unused; kept for API compatibility.

        Returns:
            pl.Expr: Polars expression that yields the hour as an integer.
        """
        return pl.col("hour").str.slice(6, 2).cast(pl.Int32)
