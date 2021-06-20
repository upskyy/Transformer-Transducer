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
from transformer_transducer.audio_encoder import AudioEncoder
from transformer_transducer.label_encoder import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerTransducer(nn.Module):
    """
    Transformer-Transducer is that every layer is identical for both audio and label encoders.
    Unlike the basic transformer structure, the audio encoder and label encoder are separate.
    So, the alignment is handled by a separate forward-backward process within the RNN-T architecture.
    And we replace the LSTM encoders in RNN-T architecture with Transformer encoders.

    Args:
        audio_encoder (AudioEncoder): Instance of audio encoder
        label_encoder (LabelEncoder): Instance of label encoder
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)

    Inputs: inputs, input_lens, targets, targets_lens
        - **inputs** (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
            `FloatTensor` of size ``(batch, dimension, seq_length)``.
        - **input_lens** (torch.LongTensor): The length of input tensor. ``(batch)``
        - **targets** (torch.LongTensor): A target sequence passed to label encoder. Typically inputs will be a padded
            `LongTensor` of size ``(batch, target_length)``
        - **targets_lens** (torch.LongTensor): The length of target tensor. ``(batch)``

    Returns: output
        - **output** (torch.FloatTensor): Result of model predictions.
    """
    def __init__(
            self,
            audio_encoder: AudioEncoder,
            label_encoder: LabelEncoder,
            num_vocabs: int,
            output_size: int = 1024,
            inner_size: int = 512,
    ) -> None:
        super(TransformerTransducer, self).__init__()
        self.audio_encoder = audio_encoder
        self.label_encoder = label_encoder
        self.joint = JointNet(num_vocabs, output_size, inner_size)

    def forward(
            self,
            inputs: Tensor,
            input_lens: Tensor,
            targets: Tensor,
            targets_lens: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `inputs, targets` for transformer transducer.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, dimension, seq_length)``.
            input_lens (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensor): A target sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            targets_lens (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            **output** (Tensor): ``(batch, seq_length, num_vocabs)``
        """
        audio_output = self.audio_encoder(inputs, input_lens)
        label_output = self.label_encoder(targets, targets_lens)

        output = self.joint(audio_output, label_output)

        return output

    @torch.no_grad()
    def decode(self, audio_outputs: Tensor, max_lens: int) -> Tensor:
        batch = audio_outputs.size(0)
        y_hats = list()

        targets = torch.LongTensor([self.label_encoder.sos_id] * batch)
        if torch.cuda.is_available():
            targets = targets.cuda()

        for i in range(max_lens):
            label_output = self.label_encoder(targets, None)
            label_output = label_output.squeeze(1)
            audio_output = audio_outputs[:, i, :]
            output = self.joint(audio_output, label_output)
            targets = output.max(1)[1]
            y_hats.append(targets)

        y_hats = torch.stack(y_hats, dim=1)

        return y_hats  # (B, T)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, inputs_lens: Tensor) -> Tensor:
        audio_outputs = self.audio_encoder(inputs, inputs_lens)
        max_lens = audio_outputs.size(1)

        return self.decode(audio_outputs, max_lens)


class JointNet(nn.Module):
    """
    Combine the audio encoder and label encoders.
    Convert them into log probability values for each word.

    Args:
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)

    Inputs: audio_encoder, label_encoder
        - **audio_encoder**: Audio encoder output
        - **label_encoder**: Label encoder output

    Returns: output
        - **output**: Tensor expressing the log probability values of each word
    """
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
            audio_encoder: Tensor,
            label_encoder: Tensor,
    ) -> Tensor:
        if audio_encoder.dim() == 3 and label_encoder.dim() == 3:  # Train
            seq_lens = audio_encoder.size(1)
            target_lens = label_encoder.size(1)

            audio_encoder = audio_encoder.unsqueeze(2)
            label_encoder = label_encoder.unsqueeze(1)

            audio_encoder = audio_encoder.repeat(1, 1, target_lens, 1)
            label_encoder = label_encoder.repeat(1, seq_lens, 1, 1)

        output = torch.cat((audio_encoder, label_encoder), dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)

        output = F.log_softmax(output, dim=-1)

        return output



