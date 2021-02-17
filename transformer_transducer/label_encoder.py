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
from transformer_transducer.mask import get_attn_pad_mask
from transformer_transducer.module import PositionalEncoding
from transformer_transducer.layer import EncoderLayer
import torch
import torch.nn as nn
import numpy as np


class LabelEncoder(nn.Module):
    def __init__(
            self,
            device: torch.device,
            num_vocabs: int,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_len: int = 5000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(LabelEncoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self_attn_mask = None
        batch = inputs.size(0)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs).to(self.device) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

        else:  # train
            inputs = inputs[inputs != self.eos_id].view(batch, -1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs).to(self.device) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

            self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, target_lens)

        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs, inputs_lens
