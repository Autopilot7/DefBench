# metrics/evaluation.py
from sklearn.metrics import accuracy_score
import torch

def evaluate_detection(detected, ground_truth):
    accuracy = accuracy_score(ground_truth, detected)
    return accuracy

def evaluate_robustness(defended_model, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = defended_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total
