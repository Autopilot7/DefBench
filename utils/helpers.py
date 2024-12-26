# utils/helpers.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

def save_model(model, path):
    """
    Lưu trạng thái của mô hình vào đường dẫn đã chỉ định.
    
    Args:
        model (torch.nn.Module): Mô hình cần lưu.
        path (str): Đường dẫn file để lưu mô hình.
    """
    torch.save(model.state_dict(), path)
    print(f"Mô hình đã được lưu tại {path}")

def load_model(model, path, device='cpu'):
    """
    Tải trạng thái của mô hình từ đường dẫn đã chỉ định.
    
    Args:
        model (torch.nn.Module): Mô hình cần tải trạng thái.
        path (str): Đường dẫn file chứa trạng thái mô hình.
        device (str): Thiết bị để tải mô hình ('cpu' hoặc 'cuda').
    
    Returns:
        torch.nn.Module: Mô hình đã được tải trạng thái.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {path}")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Mô hình đã được tải từ {path}")
    return model

def train_model(model, train_loader, criterion, optimizer, epochs=10, device='cpu', verbose=True):
    """
    Đào tạo mô hình với dữ liệu đã được nhiễm (poisoned data).
    
    Args:
        model (torch.nn.Module): Mô hình cần đào tạo.
        train_loader (DataLoader): DataLoader chứa dữ liệu đào tạo.
        criterion (torch.nn.Module): Hàm mất mát.
        optimizer (torch.optim.Optimizer): Bộ tối ưu hóa.
        epochs (int): Số lượng epoch để đào tạo.
        device (str): Thiết bị để đào tạo ('cpu' hoặc 'cuda').
        verbose (bool): Nếu True, hiển thị tiến trình đào tạo.
    
    Returns:
        torch.nn.Module: Mô hình đã được đào tạo.
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        if verbose:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        else:
            progress_bar = enumerate(train_loader)
        
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if verbose:
                progress_bar.set_postfix({'Loss': f"{running_loss/total:.4f}", 'Acc': f"{100 * correct / total:.2f}%"})
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
    
    print("Đào tạo hoàn tất")
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """
    Đánh giá mô hình trên tập dữ liệu kiểm tra.
    
    Args:
        model (torch.nn.Module): Mô hình cần đánh giá.
        test_loader (DataLoader): DataLoader chứa dữ liệu kiểm tra.
        device (str): Thiết bị để đánh giá ('cpu' hoặc 'cuda').
    
    Returns:
        float: Độ chính xác của mô hình trên tập kiểm tra.
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Đánh giá mô hình"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}%")
    return accuracy

def create_poisoned_loader(poisoned_data, batch_size=128, shuffle=True):
    """
    Tạo DataLoader từ dữ liệu đã bị nhiễm (poisoned data).
    
    Args:
        poisoned_data (list of tuples): Danh sách các cặp (images, labels) đã bị nhiễm.
        batch_size (int): Kích thước batch.
        shuffle (bool): Nếu True, xáo trộn dữ liệu.
    
    Returns:
        DataLoader: DataLoader chứa dữ liệu đã bị nhiễm.
    """
    all_images = []
    all_labels = []
    for images, labels in poisoned_data:
        all_images.append(images)
        all_labels.append(labels)
    
    # Chuyển danh sách tensor thành tensor duy nhất
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    dataset = TensorDataset(all_images, all_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_device():
    """
    Lấy thiết bị để chạy mô hình (CUDA nếu có, ngược lại CPU).
    
    Returns:
        torch.device: Thiết bị được chọn.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """
    Thiết lập seed để đảm bảo tính tái lập của các thí nghiệm.
    
    Args:
        seed (int): Giá trị seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu sử dụng nhiều GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed đã được thiết lập thành {seed}")
