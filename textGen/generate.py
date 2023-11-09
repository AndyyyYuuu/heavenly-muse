import time
import numpy
import torch
from torch import nn
from torch.utils import data
from torch import optim

best_model, char_to_int, epochs = torch.load("model/model-2-milton.pth")
print("LOADED MODEL")
print(f"Epochs trained: {epochs}")
num_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())

TEMPERATURE = 1.1

# load ascii text and covert to lowercase
filename = "paradise_lost.txt"
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

print("\n-- PROMPT --")
print(f"{prompt_txt}")
print("\n-- OUTPUT --")


with torch.no_grad():
    for i in range(gen_size):
        x = numpy.reshape(pattern, (1, len(pattern), 1)) / float(num_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        prediction = writer(x)
        # predicted_char = int_to_char[int(prediction.argmax())]

        # Model prediction to probability distribution using softmax
        prediction_probs = torch.softmax(prediction/TEMPERATURE, dim=1)
        prediction_probs = prediction_probs.squeeze().numpy()
        # Sample character index from distribution
        predicted_char_index = numpy.random.choice(len(prediction_probs), p=prediction_probs)
        # Get character by index
        predicted_char = int_to_char[predicted_char_index]

        print(predicted_char, end='')

        # Push generated character to memory
        pattern.append(int(prediction.argmax()))
        pattern.pop(0)
print("\n")