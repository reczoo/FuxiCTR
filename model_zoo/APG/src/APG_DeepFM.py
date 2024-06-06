# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
from fuxictr.pytorch.layers import FeatureEmbedding, FeatureEmbeddingDict, FactorizationMachine
from .APG import APG_MLP


class APG_DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="APG_DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hypernet_config={},
                 condition_features=[],
                 condition_mode="self-wise",
                 new_condition_emb=False,
                 rank_k=32,
                 overparam_p=1024,
                 generate_bias=True,
                 **kwargs):
        super(APG_DeepFM, self).__init__(feature_map, 
                                         model_id=model_id, 
                                         gpu=gpu,
                                         embedding_regularizer=embedding_regularizer, 
                                         net_regularizer=net_regularizer,
                                         **kwargs)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.fm = FactorizationMachine(feature_map)
        self.condition_mode = condition_mode
        self.condition_features = condition_features
        self.condition_emb_layer = None
        if condition_mode == "self-wise":
            condition_dim = None
        else:
            assert len(condition_features) > 0
            condition_dim = len(condition_features) * embedding_dim
            if new_condition_emb:
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim,
                    required_feature_columns=condition_features)
        self.mlp = APG_MLP(input_dim=feature_map.sum_emb_out_dim(),
                           output_dim=1,
                           hidden_units=hidden_units,
                           hidden_activations=hidden_activations,
                           output_activation=None,
                           dropout_rates=net_dropout,
                           batch_norm=batch_norm,
                           hypernet_config=hypernet_config,
                           condition_dim=condition_dim,
                           condition_mode=condition_mode,
                           rank_k=rank_k,
                           overparam_p=overparam_p,
                           generate_bias=generate_bias)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        condition_z = self.get_condition_z(X, feature_emb_dict)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        y_fm = self.fm(X, feature_emb)
        y_mlp = self.mlp(feature_emb.flatten(start_dim=1), condition_z)
        y_pred = y_fm + y_mlp
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_condition_z(self, X, feature_emb_dict):
        condition_z = None
        if self.condition_mode != "self-wise":
            if self.condition_emb_layer is not None:
                condition_z = self.condition_emb_layer(X, flatten_emb=True)
            else:
                condition_z = self.embedding_layer.dict2tensor(feature_emb_dict, 
                                                               feature_list=self.condition_features,
                                                               flatten_emb=True)
        return condition_z
