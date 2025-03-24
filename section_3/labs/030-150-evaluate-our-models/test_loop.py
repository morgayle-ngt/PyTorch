"""
Utilize Accuracy from torchmetrics to compute accuracy.
Be sure to set the model to evaluation model as well as set the code to not compute gradients.
"""
import torch
import torchmetrics
from load_data import test_loader

# Initialize the accuracy metric for a 2-class problem
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)

def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            accuracy_metric.update(preds, labels)
    final_accuracy = accuracy_metric.compute()
    print(f"Accuracy: {final_accuracy * 100}%")