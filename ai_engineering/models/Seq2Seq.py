import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # shape of x: (input_length, batch_size)
        outputs, hidden, cell = self.rnn(self.dropout(x))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p)

    def forward(self, x, hidden, cell):
        # x: is the embedded data from the year before
        x = self.dropout(x)
        outputs, (hidden, cell) = self.rnn(x, (hidden, cell))
        return outputs

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)

    def forward(self, x):
        return self.embedding(x)

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self,x ):
        return self.fc(x)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, embedding, mlp):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.mlp = mlp

    def forward(self, x, x_last, y_last):
        hidden, cell = self.encoder(x)
        embedding_x_last = self.embedding(x_last)
        input_decoder = torch.concatenate((hidden, cell, embedding_x_last, y_last))
        output_decoder = self.decoder(input_decoder)
        return self.mlp(output_decoder)
