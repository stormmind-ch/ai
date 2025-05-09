from models.Seq2Seq import Seq2Seq, Encoder, Decoder
from models.FNN import FNN

def get_seq2seq(hidden_size, num_layers, p):
    encoder = Encoder(8, hidden_size, num_layers, p)
    decoder = Decoder(8,hidden_size, num_layers, p)
    seq2seq = Seq2Seq(encoder, decoder)
    return seq2seq

def get_fnn():
    return FNN()