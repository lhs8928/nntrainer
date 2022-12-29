import torch
import numpy as np
import random
from transLayer_v2 import params_translated
from transLayer_v2 import optimize

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

input_file_name = "transformer_input.dat"
weight_file_name = "transformer_optimize.bin" if optimize else "transformer_origin.bin"

batch_size = 128
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 8
encoder_timestep = 150
decoder_timestep = 120
d_model = 512
dim_feedforward = 2048

class Transformer(torch.nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(Transformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.0, batch_first=True)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        src, tgt = inputs[0], inputs[1]
        output = self.transformer(src, tgt)

        loss = self.loss(output, labels[0])

        return output, loss
        

model = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

def _get_writer(file):
    def write_fn(items):
        if not isinstance(items, (list, tuple)):
            items = [items]

        for item in items:
            # np.array([item.numel()], dtype="int32").tofile(file)
            # print(item.detach().cpu().numpy())
            item.detach().cpu().numpy().tofile(file)

        return items

    return write_fn

def _rand_like(shapes, scale=1, dtype=None):
    def shape_to_np(shape, dtype=int):
        if dtype == int:
            return np.random.randint(0, 4, shape).astype(dtype=np.int32)
        else:
            return np.random.rand(*shape).astype(dtype=np.float32)

    if not isinstance(dtype, list):
        dtype = [dtype] * len(shapes)
    np_array = list([shape_to_np(s,t) for s,t in zip(shapes, dtype)])
    return list([torch.tensor(t * scale) for t in np_array])

with open(input_file_name, "wb") as f:
    write_fn = _get_writer(f)
    inputs = _rand_like([(batch_size, encoder_timestep, d_model), (batch_size, decoder_timestep, d_model)], dtype=float)
    labels = _rand_like([(batch_size, decoder_timestep, d_model)], dtype=float)
    write_fn(inputs)
    write_fn(labels)

with open(weight_file_name, "wb") as f:
    write_fn = _get_writer(f)
    write_fn(list(t for _, t in params_translated(model)))

# print(inputs)
# print(labels)
outputs, loss = model(inputs, labels)
print(outputs)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# for input, label in zip(inputs[0], labels[0]):
#     input = torch.unsqueeze(input, dim = 0)
#     label = torch.unsqueeze(label, dim = 0)
#     output, loss = model(input, label)
#     # print("input")
#     print(input)
#     # print("output")
#     print(output)
#     # print(loss)

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

