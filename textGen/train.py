

import os
import sys
import time
import numpy
import torch
from datetime import datetime
from torch import nn
from torch.utils import data
from torch import optim
from matplotlib import pyplot as plt
from tqdm import tqdm

SAVE_PATH = "model/model-3-milton.pth"

SEQ_LENGTH = 100

NUM_EPOCHS = 64
BATCH_SIZE = 32

# device = torch.device("cpu")

# Load ascii and covert to lowercase
filename = "milton_poetry_cleaned.txt"
raw_text = open(f"data/{filename}", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# Create map of char --> int
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

num_chars = len(raw_text)
num_vocab = len(chars)
print("Total Characters: ", num_chars)
print("Total Vocab: ", num_vocab)

# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, num_chars - SEQ_LENGTH, 1):
    seq_in = raw_text[i:i + SEQ_LENGTH]
    seq_out = raw_text[i + SEQ_LENGTH]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, SEQ_LENGTH, 1)
X = X / float(num_vocab)
y = torch.tensor(dataY)

# X, y = X.to(device), y.to(device)

class Poet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, num_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


model = Poet()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=BATCH_SIZE)

best_model = None
start_epoch = 1

# Load checkpoint
if os.path.exists(SAVE_PATH):
    past_state_dict = torch.load(SAVE_PATH)
    loaded_best_model, loaded_char_to_int, loaded_epoch = past_state_dict
    if loaded_epoch < NUM_EPOCHS:
        best_model = loaded_best_model
        char_to_int = loaded_char_to_int
        start_epoch = loaded_epoch+1
        print("LOADED MODEL")
        print(f"Epochs to train: {NUM_EPOCHS-start_epoch+1}")
        print(f"Save path: {SAVE_PATH}")
        input("Enter to resume training >>> ")
    else:
        print(f"Hey Andy, this model is already trained up to {NUM_EPOCHS} epochs.")
        exit(0)
else:
    print("TRAIN NEW MODEL")
    print(f"Epochs to train: {NUM_EPOCHS}")
    print(f"Save path: {SAVE_PATH}")
    input("Enter to begin training >>> ")

best_loss = numpy.inf
durations = []
print("\n*** TRAINING IN PROGRESS ***")


# Saves model at the end of each epoch
def checkpoint(best_model, char_to_int, epoch):
    torch.save([best_model, char_to_int, epoch], SAVE_PATH)


def progress_iter(it, desc):
    return tqdm(range(len(it)),
                desc=f'\t{desc}',
                unit=" batches",
                file=sys.stdout,
                colour="GREEN",
                bar_format="{desc}: {percentage:0.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]")


previous_loss = None
# Training loop
for epoch in range(start_epoch, NUM_EPOCHS+1):
    print(f"\n--- EPOCH {epoch}/{NUM_EPOCHS} at {datetime.now().strftime('%H:%M:%S')} ---")

    init_time = time.process_time()
    model.train()
    iter_len = sum(1 for _ in loader)
    percent_complete = 1
    loading_iter = iter(loader)
    for i in progress_iter(loader, "Training"):
        X_batch, y_batch = next(loading_iter)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        loading_iter = iter(loader)
        for i in progress_iter(loader, "Validating"):
            X_batch, y_batch = next(loading_iter)
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()

        # Fancy stats
        durations.append(round(time.process_time()-init_time))
        mins_left = round((sum(durations)/len(durations)*(NUM_EPOCHS-epoch))//60/5)*5
        timestamp = datetime.now().strftime("%H:%M:%S")
        # print(f"\tEpoch Completed {timestamp}")
        print(f"\tCross-Entropy Loss: {loss} ", end='')
        if previous_loss is not None:
            if loss > previous_loss:
                print(f"(+{loss-previous_loss})")
            else:
                print(f"(-{previous_loss-loss})")
        else:
            print("")
        previous_loss = loss
        print(f"\tTime Duration: {(durations[-1])//60} min, {durations[-1]%60} sec")
        print(f"\tTime Left: approx. {mins_left//60} hrs, {mins_left%60} min")

        checkpoint(best_model, char_to_int, epoch)

print("\n*** TRAINING COMPLETE ***")
print(f"Model saved as \"models/{SAVE_PATH}\"")