from transformer_transducer.model_builder import build_transformer_transducer
import torch
import warnings

warnings.filterwarnings('ignore')

batch_size = 3
seq_length = 500
input_size = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = build_transformer_transducer(
        device,
        num_vocabs=10,
        input_size=input_size,
)

inputs = torch.FloatTensor(batch_size, input_size, seq_length).to(device)
input_lengths = torch.IntTensor([500, 450, 350])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

outputs = model(inputs, input_lengths, targets)
print(outputs.size())  # torch.Size([3, 500, 9, 10])
