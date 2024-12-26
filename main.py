# main.py
import torch
import numpy as np
from datasets.mnist import get_mnist
from attacks.badnets import apply_badnets
from defenses.activation_clustering import activation_clustering_defense  # Import phương pháp phòng thủ
from metrics.evaluation import evaluate_detection, evaluate_robustness
from utils.helpers import (
    train_model, 
    save_model, 
    load_model, 
    create_poisoned_loader, 
    get_device, 
    set_seed
)
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def show_poisoned_images(poisoned_data, n=5):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        img_batch, label_batch = poisoned_data[i]
        img = img_batch[0]
        label = label_batch[0]
        img_np = img.numpy().transpose(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(img_np, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Định nghĩa mô hình đơn giản cho MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 0. Thiết lập seed và thiết bị
    set_seed(42)
    device = get_device()
    print(f"Sử dụng thiết bị: {device}")

    # 1. Tải dữ liệu
    train_loader, test_loader = get_mnist(batch_size=128)

    # 2. Tạo trigger
    trigger_pattern = np.ones((5, 5))  # Tạo pattern 5x5 làm trigger
# Khi gọi apply_badnets, truyền vị trí (0,0) là góc trái trên
    poisoned_data = apply_badnets(train_loader, trigger_pattern, target_label=0, poison_rate=0.1)

    # 3. Áp dụng tấn công BadNets
    poisoned_data = apply_badnets(train_loader, trigger_pattern, target_label=0, poison_rate=0.1)
    poisoned_train_loader = create_poisoned_loader(poisoned_data, batch_size=128, shuffle=True)

    print("Hiển thị một số ảnh đã bị tấn công...")
    show_poisoned_images(poisoned_data, n=5)

    # 4. Định nghĩa mô hình, loss và optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Đào tạo mô hình với dữ liệu đã bị nhiễm
    print("Đào tạo mô hình với dữ liệu bị nhiễm...")
    model = train_model(model, poisoned_train_loader, criterion, optimizer, epochs=5, device=device)

    # 6. Lưu mô hình đã đào tạo
    save_model(model, 'model_poisoned.pth')

    # 7. Áp dụng cơ chế phòng thủ Activation Clustering
    print("Áp dụng Activation Clustering để làm sạch dữ liệu...")
    clean_train_loader = activation_clustering_defense(model, poisoned_train_loader, target_label=0, device=device, layer='fc1')

    # 8. Đào tạo lại mô hình với dữ liệu đã được làm sạch
    print("Đào tạo lại mô hình với dữ liệu đã được làm sạch...")
    model_clean = SimpleCNN()
    criterion_clean = nn.CrossEntropyLoss()
    optimizer_clean = optim.Adam(model_clean.parameters(), lr=0.001)
    model_clean = train_model(model_clean, clean_train_loader, criterion_clean, optimizer_clean, epochs=5, device=device)

    # 9. Đánh giá mô hình sạch
    print("Đánh giá mô hình sau khi làm sạch dữ liệu...")
    accuracy_clean = evaluate_model(model_clean, test_loader, device=device)

    # 10. In kết quả
    print(f"Độ chính xác của mô hình sau khi làm sạch: {accuracy_clean:.2f}%")

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

if __name__ == "__main__":
    main()
