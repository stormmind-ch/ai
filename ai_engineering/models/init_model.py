from models.Seq2Seq import Seq2Seq, Encoder, Decoder

"""
def init_model(model:str, input_size, hidden_size, output_size, num_layers, dropout_rate=0.5):
    if model == 'VanillaNN':
        return VanillaNN(input_size,hidden_size, output_size)
    if model == 'LSTM':
        return get_lstm()
    if model == 'Seq2Seq':
        return get_seq2seq(hidden_size, num_layers)
    if model == 'LSTMAttention':
        return LSTMAttention(input_size=input_size,
                             hidden_size=hidden_size,
                             out_features=4,
                             bidirectional=True,  # try Bi-LSTM first
                             attn_type='dot')
    """


def get_seq2seq(hidden_size, num_layers, p):
    encoder = Encoder(8, hidden_size, num_layers, p)
    decoder = Decoder(8,hidden_size, num_layers, p)
    seq2seq = Seq2Seq(encoder, decoder)
    return seq2seq



