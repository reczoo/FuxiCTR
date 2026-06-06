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


from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, FactorizationMachine


class FM(BaseModel):
    """Factorization Machine (FM) model.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"FM"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        regularizer (str or None): Regularizer for embeddings and network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="FM",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 regularizer=None,
                 **kwargs):
        super(FM, self).__init__(feature_map,
                                 model_id=model_id,
                                 gpu=gpu,
                                 embedding_regularizer=regularizer,
                                 net_regularizer=regularizer,
                                 **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of FM.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.fm(X, feature_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

