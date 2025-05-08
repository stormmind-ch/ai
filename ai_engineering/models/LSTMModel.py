import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, num_layers=2, out_features=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        output, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        last_hidden = self.dropout(last_hidden)
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        return out


