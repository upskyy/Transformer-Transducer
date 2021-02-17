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

from torch import Tensor
import torch


def _get_pad_mask(inputs: Tensor, inputs_lens: Tensor):
    assert len(inputs.size()) == 3

    batch = inputs.size(0)

    pad_attn_mask = inputs.new_zeros(inputs.size()[: -1])

    for idx in range(batch):
        pad_attn_mask[idx, inputs_lens[idx]:] = 1

    return pad_attn_mask.bool()


def get_attn_pad_mask(inputs: Tensor, inputs_lens: Tensor, expand_lens):
    pad_attn_mask = _get_pad_mask(inputs, inputs_lens)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, expand_lens, 1)  # (batch, dec_T, enc_T)

    return pad_attn_mask


def _get_attn_key_pad_mask(target: Tensor, pad_id: int):
    target_lens = target.size(1)
    padding_mask = target.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, target_lens, 1)

    return padding_mask


def _get_subsequent_mask(target: Tensor):
    batch, target_lens = target.size()
    subsequent_mask = torch.triu(torch.ones((target_lens, target_lens), device=target.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch, 1, 1)

    return subsequent_mask


def get_decoder_self_attn_mask(target: Tensor, pad_id: int = 0):
    padding_mask = _get_attn_key_pad_mask(target, pad_id)
    subsequent_mask = _get_subsequent_mask(target)

    decoder_self_attn_mask = (padding_mask + subsequent_mask).bool()

    return decoder_self_attn_mask
