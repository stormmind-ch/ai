import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim : int, layer_dim: int, output_dim: int):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hc=None):
        batch_size = x.size(0)
        if hc is None:
            out, (hn, cn) = self.lstm(x)  # no hidden states passed
        else:
            h0, c0 = hc
            out, (hn, cn) = self.lstm(x, (h0, c0))

        if out.ndim == 3:
            out = out[:, -1, :]  # take last time step
        # else: assume out is already (batch_size, hidden_dim)
        out = self.fc(out)
        return out, (hn, cn)

