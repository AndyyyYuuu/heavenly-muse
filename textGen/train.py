
import time
import numpy
import torch
from torch import nn
from torch.utils import data
from torch import optim

save_path = "model-2-milton.pth"

# load ascii text and covert to lowercase
filename = "paradise_lost.txt"
raw_text = open(f"data/{filename}", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
num_chars = len(raw_text)
num_vocab = len(chars)
print("Total Characters: ", num_chars)
print("Total Vocab: ", num_vocab)


# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, num_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(num_vocab)
y = torch.tensor(dataY)

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


n_epochs = 32
batch_size = 32
model = Poet()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = numpy.inf
durations = []
print("\n*** TRAINING IN PROGRESS ***")
for epoch in range(n_epochs):
    init_time = time.process_time()
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        durations.append(round(time.process_time()-init_time))
        mins_left = round((sum(durations)/len(durations)*(n_epochs-epoch-1))//60/5)*5
        print(f"\n-< EPOCH {epoch} >-")
        print(f"Cross-Entropy Loss: {loss}")
        print(f"Time Duration: {(durations[-1])//60} min, {durations[-1]%60} sec")
        print(f"Time Left: approx. {mins_left//60} hrs, {mins_left%60} min")
print("\n*** TRAINING COMPLETE ***")
torch.save([best_model, char_to_int], f"models/{save_path}")
print(f"Model saved as \"models/{save_path}\"")