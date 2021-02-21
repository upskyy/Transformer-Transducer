# Transformer-Transducer 
Transformer-Transducer is that every layer is identical for both audio and label encoders.
Unlike the basic transformer structure, the audio encoder and label encoder are separate.
So, the alignment is handled by a separate forward-backward process within the RNN-T architecture.
And we replace the LSTM encoders in RNN-T architecture with Transformer encoders.

PyTorch Implementation of [Transformer Transducer](https://arxiv.org/abs/2002.02562)  


![b](https://user-images.githubusercontent.com/54731898/108169684-6156ff80-713c-11eb-9469-80820d91c365.PNG)   

There are an Audio encoder and a Label encoder structure.  
I appreciate [any feedback or contribution](https://github.com/hasangchun).  


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

## Author
- seomk9896@naver.com
