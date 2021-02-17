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
input_lengths = torch.IntTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30]).to(device)

model = build_transformer_transducer(
        device,
        num_vocabs=10,
        input_size=input_size,
)

outputs = model.recognize(inputs, input_lengths)

print(outputs.size())  # torch.Size([4, 500])
