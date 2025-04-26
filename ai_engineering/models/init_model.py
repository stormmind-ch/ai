from models.VanillaNNModel import VanillaNN
from models.LSTMModel import LSTM

def init_model(model:str, input_size, hidden_size, output_size):
    if model == 'VanillaNN':
        return VanillaNN(input_size,hidden_size, output_size)
    if model == 'LSTM':
        return LSTM(input_size, hidden_size, hidden_size, output_size)
