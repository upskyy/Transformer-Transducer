# Copyright (c) 2021, Sangchun Ha. All rights reserved.
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

from typing import Optional, Tuple
from torch import Tensor
from transformer_transducer.attention import MultiHeadAttention
from transformer_transducer.module import PositionWiseFeedForward
import torch.nn as nn


class EncoderLayer(nn.Module):
    """
    Repeated layers common to audio encoders and label encoders

    Args:
        model_dim (int): the number of features in the encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of encoder layer (default: 0.1)

    Inputs: inputs, self_attn_mask
        - **inputs**: Audio feature or label feature
        - **self_attn_mask**: Self attention mask to use in multi-head attention

    Returns: outputs, attn_distribution
        - **outputs**: Tensor containing higher (audio, label) feature values
        - **attn_distribution**: Attention distribution in multi-head attention
    """
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_heads: int = 8,
            dropout: float = 0.1,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)

    def forward(
            self,
            inputs: Tensor,
            self_attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs : A input sequence passed to encoder layer. ``(batch, seq_length, dimension)``
            self_attn_mask : Self attention mask to cover up padding ``(batch, seq_length, seq_length)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            **attn_distribution** (Tensor): ``(batch, seq_length, seq_length)``
        """
        inputs = self.layer_norm(inputs)
        self_attn_output, attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output += inputs

        self_attn_output = self.layer_norm(self_attn_output)
        ff_output = self.feed_forward(self_attn_output)
        output = self.encoder_dropout(ff_output + self_attn_output)

        return output, attn_distribution
