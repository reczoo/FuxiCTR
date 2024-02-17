# =========================================================================
# Copyright (C) 2024, FuxiCTR Authors. All rights reserved.
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


from .h5_block_dataloader import H5BlockDataLoader
from .h5_dataloader import H5DataLoader


class DataLoader(H5DataLoader, H5BlockDataLoader):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, verbose=0, streaming=False, **kwargs):
        if streaming:
            H5BlockDataLoader.__init__(self, feature_map, stage, train_data, valid_data, test_data,
                                       batch_size, shuffle, verbose, **kwargs)
        else:
            H5DataLoader.__init__(self, feature_map, stage, train_data, valid_data, test_data,
                                  batch_size, shuffle, **kwargs)
