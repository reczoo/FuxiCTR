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


from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import LogisticRegression


class LR(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="LR", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 regularizer=None, 
                 **kwargs):
        super(LR, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer, 
                                 **kwargs)
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        y_pred = self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

