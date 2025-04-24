from models.VanillaNN.train_VanillaNN import train as train_vanilla
from models.LSTM.train_LSTM import train_and_validate as train_LSTM
from models.LSTM.train_LSTM import validate as validata_LSTM
from models.VanillaNN.VanillaNNModel import VanillaNN
from models.LSTM.LSTMModel import LSTM
from models.VanillaNN.validate_VanillaNN import validate as validate_vanilla

def train(model, train_loader, val_loader, criterion, optimizer,epochs, device):
    if type(model) is VanillaNN:
        train_vanilla(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    if type(model) is LSTM:
        train_LSTM(model, train_loader, val_loader, criterion, optimizer, epochs, device)

def validate(model, test_loader, criterion, device):
    if type(model) is VanillaNN:
        return validate_vanilla(model, test_loader, criterion, device)
    if type(model) is LSTM:
        return validata_LSTM(model, test_loader, criterion, device)