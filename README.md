# Transformer-Transducer
PyTorch Implementation of [Transformer Transducer](https://arxiv.org/abs/2002.02562)  


![b](https://user-images.githubusercontent.com/54731898/108169684-6156ff80-713c-11eb-9469-80820d91c365.PNG)   

It is an Audio encoder and a Label encoder structure.  
I appreciate any feedback or contribution.  
```
from transformer_transducer.model_builder import build_transformer_transducer
import torch
import warnings

warnings.filterwarnings('ignore')

batch_size = 4
seq_length = 500
input_size = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.FloatTensor(batch_size, input_size, seq_length).to(device)
input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30]).to(device)

model = build_transformer_transducer(
        device,
        num_vocabs=10,
        input_size=input_size,
)

outputs = model.recognize(inputs, input_lengths)
```

## Reference
- [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562)

## Author
- seomk9896@naver.com
