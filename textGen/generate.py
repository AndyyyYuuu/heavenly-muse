import time
import numpy
import torch
from torch import nn
from torch.utils import data
from torch import optim

best_model, char_to_int = torch.load("model/model-1-hemingway.pth")
num_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())

# load ascii text and covert to lowercase
filename = "the_sun_also_rises.txt"
raw_text = open(f"data/{filename}", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

prompt_size = 100
gen_size = 1000
rand_start = numpy.random.randint(0, len(raw_text)-prompt_size)
prompt_txt = raw_text[rand_start:rand_start+prompt_size]
pattern = [char_to_int[c] for c in prompt_txt]

class Poet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, num_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

writer = Poet()
writer.load_state_dict(best_model)
writer.eval()

print(f"Prompt: {prompt_txt}")
with torch.no_grad():
    for i in range(gen_size):
        x = numpy.reshape(pattern, (1, len(pattern), 1)) / float(num_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        prediction = writer(x)
        predicted_char = int_to_char[int(prediction.argmax())]
        print(predicted_char, end='')

        pattern.append(int(prediction.argmax()))
        pattern.pop(0)
