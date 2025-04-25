import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim : int, layer_dim: int, output_dim: int):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            # h0 is the hidden state
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            # c state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn
