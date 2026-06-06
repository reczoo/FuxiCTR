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


import torch
from torch import nn
from .dot_product_attention import ScaledDotProductAttention
from ..activations import Dice
from ..blocks.mlp_block import MLP_Block


class DIN_Attention(nn.Module):
    """DIN-style target attention that computes attention weights over a history sequence.

    ``DIN_Attention`` concatenates the target item with each item in the history sequence
    using element-wise difference and product, then feeds the result into an MLP to
    produce attention scores.

    Args:
        embedding_dim (int, optional): Dimensionality of item embeddings. Default: ``64``.
        attention_units (list[int], optional): Hidden units of the attention MLP. Default: ``[32]``.
        hidden_activations (str or list, optional): Activation(s) for hidden layers.
            Default: ``"ReLU"``.
        output_activation (str, optional): Activation for the output layer. Default: ``None``.
        dropout_rate (float, optional): Dropout rate for the attention MLP. Default: ``0``.
        batch_norm (bool, optional): Whether to apply batch normalization. Default: ``False``.
        use_softmax (bool, optional): Whether to apply softmax over attention weights.
            Default: ``False``.
    """

    def __init__(self,
                 embedding_dim=64,
                 attention_units=[32],
                 hidden_activations="ReLU",
                 output_activation=None,
                 dropout_rate=0,
                 batch_norm=False,
                 use_softmax=False):
        super(DIN_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
            hidden_activations = [Dice(units) for units in attention_units]
        self.attention_layer = MLP_Block(input_dim=4 * embedding_dim,
                                         output_dim=1,
                                         hidden_units=attention_units,
                                         hidden_activations=hidden_activations,
                                         output_activation=output_activation,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm)

    def forward(self, target_item, history_sequence, mask=None):
        """Compute the DIN attention-weighted sum of the history sequence.

        Args:
            target_item (torch.Tensor): Target item embedding of shape ``(batch_size, emb)``.
            history_sequence (torch.Tensor): History sequence embeddings of shape
                ``(batch_size, seq_len, emb)``.
            mask (torch.Tensor, optional): Mask of ``history_sequence`` where 0 indicates
                masked positions. Default: ``None``.

        Returns:
            torch.Tensor: Aggregated history representation of shape ``(batch_size, emb)``.
        """
        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([target_item, history_sequence, target_item - history_sequence,
                                     target_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        if self.use_softmax:
            if mask is not None:
                attention_weight += -1.e9 * (1 - mask.float())
            attention_weight = attention_weight.softmax(dim=-1)
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
        return output


class MultiHeadTargetAttention(nn.Module):
    """Multi-head target attention for sequential recommendation.

    ``MultiHeadTargetAttention`` projects the target item and history sequence into
    multiple attention heads, applies scaled dot-product attention, and concatenates
    the results.

    Args:
        input_dim (int, optional): Dimensionality of input embeddings. Default: ``64``.
        attention_dim (int, optional): Total dimensionality of the attention space.
            Default: ``64``.
        num_heads (int, optional): Number of attention heads. Default: ``1``.
        dropout_rate (float, optional): Dropout rate for attention weights. Default: ``0``.
        use_scale (bool, optional): Whether to scale dot products by ``sqrt(head_dim)``.
            Default: ``True``.
        use_qkvo (bool, optional): Whether to use linear projections for Q, K, V, and O.
            If False, ``attention_dim`` is set to ``input_dim``. Default: ``True``.
    """

    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True):
        super(MultiHeadTargetAttention, self).__init__()
        if not use_qkvo:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, target_item, history_sequence, mask=None):
        """Compute multi-head target attention over the history sequence.

        Args:
            target_item (torch.Tensor): Target item embedding of shape ``(batch_size, emb)``.
            history_sequence (torch.Tensor): History sequence embeddings of shape
                ``(batch_size, seq_len, emb)``.
            mask (torch.Tensor, optional): Mask of ``history_sequence`` where 0 indicates
                masked positions. Default: ``None``.

        Returns:
            torch.Tensor: Aggregated representation of shape ``(batch_size, input_dim)``.
        """
        # linear projection
        if self.use_qkvo:
            query = self.W_q(target_item)
            key = self.W_k(history_sequence)
            value = self.W_v(history_sequence)
        else:
            query, key, value = target_item, history_sequence, history_sequence

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        return output
