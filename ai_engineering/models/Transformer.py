from torch import nn
import torch
class TransformerForecast(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, seq_length):
        super(TransformerForecast, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:x.size(1)]
        transformer_out = self.transformer(x, x)
        return self.fc_out(transformer_out[:, -1, :])