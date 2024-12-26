# main.py

import torch
import numpy as np
from datasets.mnist import get_mnist
from attacks.badnets import apply_badnets
from defenses.activation_clustering import activation_clustering_defense
from defenses.fine_pruning import fine_pruning_defense
from defenses.strip import strip_defense  # Import STRIP defense
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

from metrics.evaluation import evaluate_model, evaluate_asr, evaluate_target_label_accuracy
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
import argparse
import sys
import os

# Định nghĩa một mapping từ tên phương pháp phòng thủ tới hàm hoặc lớp tương ứng
defense_methods = {
    'activation_clustering': activation_clustering_defense,
    'fine_pruning': fine_pruning_defense,
    'strip': strip_defense,  # Thêm STRIP vào danh sách
    # 'neural_cleanse': neural_cleanse_defense,
    # Thêm các phương pháp phòng thủ khác ở đây
}

# Định nghĩa mô hình SimpleCNN
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

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Backdoor Defense Mechanisms')
    parser.add_argument('--defense', type=str, required=True, choices=defense_methods.keys(),
                        help='Defense method to use: ' + ', '.join(defense_methods.keys()))
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--poison_rate', type=float, default=0.1, help='Poison rate for backdoor attack')
    parser.add_argument('--target_label', type=int, default=0, help='Target label for backdoor attack')
    parser.add_argument('--layer_name', type=str, default='fc1', help='Layer name for defense methods that require it')
    parser.add_argument('--prune_ratio', type=float, default=0.2, help='Prune ratio for Fine-Pruning')
    parser.add_argument('--fine_tune_epochs', type=int, default=5, help='Fine-tune epochs for Fine-Pruning')
    parser.add_argument('--strip_alpha', type=float, default=1.0, help='Alpha for STRIP defense')
    parser.add_argument('--strip_N', type=int, default=64, help='Number of clean samples to mix with each test sample in STRIP')
    parser.add_argument('--strip_defense_fpr', type=float, default=0.05, help='False Positive Rate threshold for STRIP defense')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--save_path', type=str, default='./models',
                        help='Path to save the trained models and results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Thiết lập seed ngẫu nhiên
    set_seed(args.seed)
    
    # Thiết lập thiết bị
    device = args.device
    print(f"Using device: {device}")
    
    # Tải dữ liệu MNIST
    train_loader, test_loader = get_mnist(batch_size=args.batch_size)
    
    # 2. Tạo trigger
    trigger_pattern = np.ones((5, 5))  # Tạo pattern 5x5 làm trigger
    # Khi gọi apply_badnets, truyền vị trí (0,0) là góc trái trên
    poisoned_data = apply_badnets(train_loader, trigger_pattern, target_label=0, poison_rate=0.1)
    poisoned_train_loader = create_poisoned_loader(poisoned_data, batch_size=128, shuffle=True)

    print("Hiển thị một số ảnh đã bị tấn công...")
    show_poisoned_images(poisoned_data, n=20)
    
    # Áp dụng tấn công BadNets vào tập kiểm tra để tạo poisoned_test_loader
    print("Applying BadNets attack to testing data for ASR evaluation...")
    poisoned_data_test = apply_badnets(test_loader, trigger_pattern, target_label=args.target_label, poison_rate=args.poison_rate)
    poisoned_test_loader = create_poisoned_loader(poisoned_data_test, batch_size=args.batch_size, shuffle=False)
    
    # Định nghĩa mô hình, hàm mất mát và optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Đào tạo mô hình trên dữ liệu bị nhiễm
    print("Training model on poisoned training data...")
    model = train_model(model, poisoned_train_loader, criterion, optimizer, epochs=args.epochs, device=device)
    
    # Lưu mô hình đã được đào tạo
    os.makedirs(args.save_path, exist_ok=True)
    poisoned_model_path = os.path.join(args.save_path, 'model_poisoned.pth')
    save_model(model, poisoned_model_path)
    print(f"Poisoned model saved at {poisoned_model_path}")
    
    # Đánh giá mô hình trước khi áp dụng phòng thủ
    print("Evaluating poisoned model before defense...")
    acc_clean_before = evaluate_model(model, test_loader, device=device)
    asr_before = evaluate_asr(model, poisoned_test_loader, target_label=args.target_label, device=device)
    target_label_acc_before = evaluate_target_label_accuracy(model, test_loader, target_label=args.target_label, device=device)
    
    print(f"Before Defense - Test Accuracy (Clean): {acc_clean_before:.2f}%")
    print(f"Before Defense - Attack Success Rate (ASR): {asr_before:.2f}%")
    print(f"Before Defense - Target Label Accuracy on Clean Data: {target_label_acc_before:.2f}%")
    
    # Chọn phương pháp phòng thủ
    defense_name = args.defense
    defense_method = defense_methods[defense_name]
    
    print(f"Applying defense: {defense_name}")
    
    if defense_name == 'activation_clustering':
        # Áp dụng Activation Clustering
        print("Applying Activation Clustering Defense...")
        clean_train_loader = defense_method(
            model=model,
            train_loader=poisoned_train_loader,
            target_label=args.target_label,
            device=device,
            layer_name=args.layer_name
        )
        # Đào tạo lại mô hình trên dữ liệu đã được làm sạch
        print("Training a new model on cleaned data after Activation Clustering...")
        model_clean = SimpleCNN()
        criterion_clean = nn.CrossEntropyLoss()
        optimizer_clean = optim.Adam(model_clean.parameters(), lr=args.lr)
        model_clean = train_model(model_clean, clean_train_loader, criterion_clean, optimizer_clean, epochs=args.epochs, device=device)
        # Đánh giá mô hình đã làm sạch
        acc_clean_after = evaluate_model(model_clean, test_loader, device=device)
        asr_after = evaluate_asr(model_clean, poisoned_test_loader, target_label=args.target_label, device=device)
        target_label_acc_after = evaluate_target_label_accuracy(model_clean, test_loader, target_label=args.target_label, device=device)
        print(f"After Defense - Test Accuracy (Clean): {acc_clean_after:.2f}%")
        print(f"After Defense - Attack Success Rate (ASR): {asr_after:.2f}%")
        print(f"After Defense - Target Label Accuracy on Clean Data: {target_label_acc_after:.2f}%")
        # Lưu mô hình đã làm sạch
        clean_model_path = os.path.join(args.save_path, 'model_activation_cleaned.pth')
        save_model(model_clean, clean_model_path)
        print(f"Cleaned model saved at {clean_model_path}")
    
    elif defense_name == 'fine_pruning':
        # Áp dụng Fine-Pruning
        print("Applying Fine-Pruning Defense...")
        metrics_fine_pruning = defense_method(
            model=model,
            train_loader_clean=poisoned_train_loader,  # Trong thực tế, nên sử dụng DataLoader chứa dữ liệu sạch
            test_loader_clean=test_loader,
            test_loader_bd=poisoned_test_loader,  # Sử dụng poisoned_test_loader để đánh giá ASR
            device=device,
            layer_name=args.layer_name,
            prune_ratio=args.prune_ratio,
            fine_tune_epochs=args.fine_tune_epochs,
            save_path=args.save_path
        )
        # Đánh giá mô hình sau Fine-Pruning
        acc_pruned = metrics_fine_pruning['finetuned']['acc_clean']
        asr_pruned = metrics_fine_pruning['finetuned']['asr']
        target_label_acc_pruned = metrics_fine_pruning['finetuned']['target_label_acc']
        print(f"After Defense - Test Accuracy (Clean): {acc_pruned:.2f}%")
        print(f"After Defense - Attack Success Rate (ASR): {asr_pruned:.2f}%")
        print(f"After Defense - Target Label Accuracy on Clean Data: {target_label_acc_pruned:.2f}%")
    
    elif defense_name == 'strip':
        # Áp dụng STRIP
        print("Applying STRIP Defense...")
        metrics_strip = defense_method(
            model=model,
            clean_loader=test_loader,             # Dữ liệu sạch để xác định ngưỡng
            poisoned_loader=poisoned_test_loader,  # Dữ liệu bị nhiễm để đánh giá ASR
            device=device,
            strip_alpha=args.strip_alpha,
            N=args.strip_N,
            defense_fpr=args.strip_defense_fpr,
            save_path=args.save_path
        )
        # In các chỉ số đánh giá
        print("STRIP Defense Evaluation Metrics:")
        print(f"Confusion Matrix: TN={metrics_strip['confusion_matrix']['TN']}, "
              f"FP={metrics_strip['confusion_matrix']['FP']}, "
              f"FN={metrics_strip['confusion_matrix']['FN']}, "
              f"TP={metrics_strip['confusion_matrix']['TP']}")
        print(f"TPR (Recall): {metrics_strip['metrics']['TPR']:.4f}")
        print(f"FPR: {metrics_strip['metrics']['FPR']:.4f}")
        print(f"Precision: {metrics_strip['metrics']['Precision']:.4f}")
        print(f"Accuracy: {metrics_strip['metrics']['Accuracy']:.4f}")
        print(f"F1 Score: {metrics_strip['metrics']['F1_Score']:.4f}")
    
    else:
        raise ValueError(f"Defense method '{defense_name}' is not implemented.")
    
    print("Defense process completed successfully.")

if __name__ == "__main__":
    main()
