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

from typing import Tuple
from torch import Tensor
from transformer_transducer.attention import MultiHeadAttention
from transformer_transducer.mask import (
    get_attn_pad_mask,
    get_decoder_self_attn_mask,
)
from transformer_transducer.module import (
    PositionWiseFeedForward,
    PositionalEncoding,
)
import torch
import torch.nn as nn
import numpy as np


class Decoder(nn.Module):
    def __init__(
            self,
            device: torch.device,
            num_vocabs: int,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_len: int = 5000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            encoder_output_lens: Tensor
    ) -> Tensor:
        batch = inputs.size(0)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
        else:  # train
            inputs = inputs[inputs != self.eos_id].view(batch, -1)

        target_lens = inputs.size(1)

        enc_dec_attn_mask = get_attn_pad_mask(encoder_output, encoder_output_lens, target_lens)
        self_attn_mask = get_decoder_self_attn_mask(inputs, self.pad_id)

        embedding_output = self.embedding(inputs).to(self.device) * self.scale
        positional_encoding_output = self.positional_encoding(target_lens)

        inputs = embedding_output + positional_encoding_output
        outputs = self.input_dropout(inputs)

        for decoder_layer in self.decoder_layers:
            outputs, _, _ = decoder_layer(outputs, encoder_output, self_attn_mask, enc_dec_attn_mask)

        return outputs


class DecoderLayer(nn.Module):
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_heads: int = 8,
            dropout: float = 0.1,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.decoder_dropout = nn.Dropout(p=dropout)
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)

    def forward(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            self_attn_mask: Tensor,
            enc_dec_attn_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        inputs = self.layer_norm(inputs)
        self_attn_output, self_attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output += inputs

        self_attn_output = self.layer_norm(self_attn_output)
        enc_dec_attn_output, enc_dec_attn_distribution = self.encoder_decoder_attention(self_attn_output,
                                                                                        encoder_output,
                                                                                        encoder_output,
                                                                                        enc_dec_attn_mask)
        enc_dec_attn_output += self_attn_output

        enc_dec_attn_output = self.layer_norm(enc_dec_attn_output)
        ff_output = self.feed_forward(enc_dec_attn_output)
        output = self.decoder_dropout(ff_output + enc_dec_attn_output)

        return output, self_attn_distribution, enc_dec_attn_distribution
