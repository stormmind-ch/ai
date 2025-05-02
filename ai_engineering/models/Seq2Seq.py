import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p, batch_first=True)

    def forward(self, x):
        # shape of x: {batch_size, n_sequences, n_features}
        outputs, (hidden, cell) = self.rnn(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, hidden, cell):
        # shape of x: {batch_size, n_sequences, n_features}: This is the data of the previous year.
        outputs, (hidden, cell) = self.rnn(x, (hidden, cell)) # outputs shape: (batch_size, seq_len, hidden_size)
        last_output = outputs[:, -1, :]
        pred = self.fc(last_output)
        return pred

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder : Decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_current, x_previous):
        hidden, cell = self.encoder(x_current)
        decoder_output = self.decoder(x_previous, hidden, cell)
        return decoder_output
