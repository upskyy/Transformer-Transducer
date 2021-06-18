# Transformer-Transducer

**Transformer-Transducer** is that every layer is identical for both audio and label encoders.
Unlike the basic transformer structure, the audio encoder and label encoder are separate.
So, the alignment is handled by a separate forward-backward process within the RNN-T architecture.
And this replace the LSTM encoders with Transformer encoders in RNN-T architecture.



This repository contains only model code, but you can train with transformer transducer at [openspeech](https://github.com/sooftware/openspeech).


          
## Usage
```python
from transformer_transducer.model_builder import build_transformer_transducer
import torch

BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, NUM_VOCABS = 3, 500, 80, 10

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = build_transformer_transducer(
        device,
        num_vocabs=NUM_VOCABS,
        input_size=INPUT_SIZE,
)

inputs = torch.FloatTensor(BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH).to(device)
input_lengths = torch.IntTensor([500, 450, 350])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

outputs = model(inputs, input_lengths, targets, target_lengths)
```

## Reference
- [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562)  
  
## License
```
Copyright 2021 Sangchun Ha.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```  
