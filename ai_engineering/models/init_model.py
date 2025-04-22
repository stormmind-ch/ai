from models.VanillaNN import VanillaNN


def init_model(model:str, input_size, hidden_size, output_size):
    if model == 'VanillaNN':
        return VanillaNN(input_size,hidden_size, output_size)
