from models.FNN import FNN
from models.Transformer import TransformerForecast

def get_model(model: str):
    if model == 'FNN':
        return FNN()
    if model == 'TRANSFORMER':
        return TransformerForecast(8, 256, 16, 6, 2, 13)
    raise ValueError(f"No model with name {model} found.")