# metrics/evaluation.py

import torch
from tqdm import tqdm

def evaluate_model(model, dataloader, device='cpu'):
    """
    Đánh giá mô hình trên tập dữ liệu kiểm tra và tính độ chính xác tổng thể.
    
    Args:
        model (torch.nn.Module): Mô hình cần đánh giá.
        dataloader (DataLoader): DataLoader chứa dữ liệu kiểm tra.
        device (str): Thiết bị để chạy đánh giá ('cpu' hoặc 'cuda').
    
    Returns:
        float: Độ chính xác tổng thể trên tập kiểm tra (theo phần trăm).
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Đánh giá mô hình"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_asr(model, dataloader, target_label, device='cpu'):
    """
    Đánh giá Attack Success Rate (ASR) trên tập dữ liệu kiểm tra backdoor.
    
    ASR là tỷ lệ phần trăm các mẫu bị nhiễm được phân loại thành nhãn mục tiêu.
    
    Args:
        model (torch.nn.Module): Mô hình cần đánh giá.
        dataloader (DataLoader): DataLoader chứa dữ liệu kiểm tra backdoor.
        target_label (int): Nhãn mục tiêu của backdoor.
        device (str): Thiết bị để chạy đánh giá ('cpu' hoặc 'cuda').
    
    Returns:
        float: ASR (theo phần trăm).
    """
    model.to(device)
    model.eval()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Đánh giá ASR"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target_label).sum().item()
            total += inputs.size(0)
    
    asr = 100 * correct / total
    return asr

def evaluate_target_label_accuracy(model, dataloader, target_label, device='cpu'):
    """
    Đánh giá độ chính xác trên các mẫu có nhãn mục tiêu trong tập dữ liệu kiểm tra.
    
    Args:
        model (torch.nn.Module): Mô hình cần đánh giá.
        dataloader (DataLoader): DataLoader chứa dữ liệu kiểm tra.
        target_label (int): Nhãn mục tiêu.
        device (str): Thiết bị để chạy đánh giá ('cpu' hoặc 'cuda').
    
    Returns:
        float: Độ chính xác trên các mẫu có nhãn mục tiêu (theo phần trăm).
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Đánh giá độ chính xác trên nhãn mục tiêu"):
            inputs, labels = inputs.to(device), labels.to(device)
            mask = (labels == target_label)
            if mask.sum().item() == 0:
                continue  # Không có mẫu nào với nhãn mục tiêu trong batch này
            inputs_target = inputs[mask]
            labels_target = labels[mask]
            outputs = model(inputs_target)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_target.size(0)
            correct += (predicted == labels_target).sum().item()
    
    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy
