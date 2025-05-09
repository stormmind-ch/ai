import torch
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from tqdm import tqdm
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import classification_report

def validate(model, dataloader, criterion, device, testing=False):
    """
    Args:
        model: the model which should be validated
        dataloader: dataloader containing the validation dataset
        criterion: Pytorch criterion, e.g. CrossEntropyLoss
        device: device to run on

    Returns:
        avg_loss: average loss over the whole test set
        accuracy
        precision
        specificity
        f1
        true labels
        predictions
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", unit="batch", file=sys.stdout, dynamic_ncols=True):
            inputs = inputs[:, :, :3]
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs= model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    if testing:
        log_classification_report_heatmap(all_labels, all_preds)
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds, normalize=True)
    precision = precision_score(all_labels, all_preds, average='macro')
    specificity = specificity_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, precision, specificity, f1, all_labels, all_preds



def log_classification_report_heatmap(all_labels, all_preds):
    report = classification_report(all_labels, all_preds, output_dict=True)
    report_df = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(report_df[["precision", "recall", "f1-score"]].astype(float), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    wandb.log({"classification_report_heatmap": wandb.Image(plt)})
    plt.close()
