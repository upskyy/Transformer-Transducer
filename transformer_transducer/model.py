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
from transformer_transducer.encoder import Encoder
from transformer_transducer.decoder import Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointNet(nn.Module):
    def __init__(
            self,
            num_vocabs: int,
            output_size: int = 1024,
            inner_size: int = 512,
    ) -> None:
        super(JointNet, self).__init__()
        self.fc1 = nn.Linear(output_size, inner_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(inner_size, num_vocabs)

    def forward(
            self,
            encoder_output: Tensor,
            decoder_output: Tensor,
    ) -> Tensor:
        if encoder_output.dim() == 3 and decoder_output.dim() == 3:  # Train
            seq_lens = encoder_output.size(1)
            target_lens = decoder_output.size(1)

            encoder_output = encoder_output.unsqueeze(2)
            decoder_output = decoder_output.unsqueeze(1)

            encoder_output = encoder_output.repeat(1, 1, target_lens, 1)
            decoder_output = decoder_output.repeat(1, seq_lens, 1, 1)

        output = torch.cat((encoder_output, decoder_output), dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)

        return output.log_softmax(dim=-1)


class TransformerTransducer(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            num_vocabs: int,
            output_size: int = 1024,
            inner_size: int = 512,
    ) -> None:
        super(TransformerTransducer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = JointNet(num_vocabs, output_size, inner_size)

    def forward(
            self,
            inputs: Tensor,
            input_lens: Tensor,
            targets: Tensor,
    ) -> Tensor:
        encoder_output, encoder_output_lens = self.encoder(inputs, input_lens)
        decoder_output = self.decoder(targets, encoder_output, encoder_output_lens)

        output = self.joint(encoder_output, decoder_output)

        return output

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, encoder_output_lens: Tensor, max_lens: int) -> Tensor:
        batch = encoder_outputs.size(0)
        y_hats = list()

        inputs = torch.LongTensor([self.decoder.sos_id] * batch)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        for i in range(max_lens):
            dec_out = self.decoder(inputs, encoder_outputs, encoder_output_lens)
            dec_out = dec_out.squeeze(1)
            enc_out = encoder_outputs[:, i, :]
            output = self.joint(enc_out, dec_out)
            inputs = output.max(1)[1]
            y_hats.append(inputs)

        y_hats = torch.stack(y_hats, dim=1)

        return y_hats  # (B, T)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, inputs_lens: Tensor) -> Tensor:
        encoder_outputs, encoder_output_lens = self.encoder(inputs, inputs_lens)
        max_lens = encoder_outputs.size(1)

        return self.decode(encoder_outputs, encoder_output_lens, max_lens)
