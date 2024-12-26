# defenses/fine_pruning.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

from utils.helpers import save_model, load_model, train_model, evaluate_model, get_device, create_poisoned_loader
from metrics.evaluation import evaluate_model, evaluate_asr, evaluate_target_label_accuracy

def fine_pruning_defense(model, train_loader_clean, test_loader_clean, test_loader_bd, target_label, device='cpu', layer_name='fc1', prune_ratio=0.2, fine_tune_epochs=5, save_path='./models'):
    """
    Áp dụng phương pháp Fine-Pruning để phát hiện và loại bỏ các mẫu bị nhiễm.

    Args:
        model (torch.nn.Module): Mô hình đã được đào tạo với dữ liệu bị nhiễm.
        train_loader_clean (DataLoader): DataLoader chứa dữ liệu đào tạo sạch.
        test_loader_clean (DataLoader): DataLoader chứa dữ liệu kiểm tra sạch.
        test_loader_bd (DataLoader): DataLoader chứa dữ liệu kiểm tra backdoor.
        device (str): Thiết bị ('cpu' hoặc 'cuda').
        layer_name (str): Tên của lớp để trích xuất kích hoạt.
        prune_ratio (float): Tỷ lệ phần trăm neuron cần cắt tỉa.
        fine_tune_epochs (int): Số lượng epoch để tinh chỉnh.
        save_path (str): Đường dẫn để lưu mô hình đã được cắt tỉa và tinh chỉnh.

    Returns:
        dict: Các chỉ số đánh giá sau Fine-Pruning và Fine-Tuning.
    """
    model = model.to(device)
    model.eval()

    activations = []
    labels_list = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Đăng ký hook
    hook = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            print(f"Hook registered on layer: {layer_name}")
            break

    if hook is None:
        raise ValueError(f"Lớp {layer_name} không tồn tại trong mô hình.")

    # Forward pass để thu thập kích hoạt
    print("Collecting activations for pruning...")
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader_clean, desc="Forwarding data to collect activations"):
            inputs = inputs.to(device)
            _ = model(inputs)

    # Gỡ bỏ hook
    hook.remove()

    # Chuyển đổi danh sách thành mảng numpy
    activations = np.concatenate(activations, axis=0)
    mean_activations = activations.mean(axis=0)  # Mean activation per neuron

    # Xếp hạng neuron theo kích hoạt trung bình
    ranked_indices = np.argsort(mean_activations)  # Tăng dần
    num_neurons = len(ranked_indices)
    num_prune = int(prune_ratio * num_neurons)
    prune_indices = ranked_indices[:num_prune]
    print(f"Pruning {num_prune} out of {num_neurons} neurons.")

    # Lấy lớp mục tiêu để cắt tỉa
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    if target_layer is None:
        raise ValueError(f"Lớp {layer_name} không tồn tại trong mô hình.")

    # Tạo mask cho cắt tỉa
    prune_mask = torch.ones_like(target_layer.weight.data)
    prune_mask[:, prune_indices] = 0  # Cắt tỉa các neuron này

    # Áp dụng cắt tỉa
    prune.custom_from_mask(target_layer, name='weight', mask=prune_mask.to(device))
    print(f"Pruned neurons indices: {prune_indices}")

    # Đánh giá mô hình đã prune
    print("Evaluating pruned model...")
    acc_clean = evaluate_model(model, test_loader_clean, device=device)
    # python
    asr = evaluate_asr(model, test_loader_bd, target_label=target_label, device=device)
    target_label_acc = evaluate_target_label_accuracy(model, test_loader_clean, target_label=target_label, device=device)

    print(f"Pruned Model - Test Accuracy (Clean): {acc_clean:.2f}%")
    print(f"Pruned Model - Attack Success Rate (ASR): {asr:.2f}%")
    print(f"Pruned Model - Target Label Accuracy on Clean Data: {target_label_acc:.2f}%")

    # Lưu mô hình đã được prune
    pruned_model_path = os.path.join(save_path, 'model_fine_pruned.pth')
    os.makedirs(save_path, exist_ok=True)
    save_model(model, pruned_model_path)
    print(f"Pruned model saved at {pruned_model_path}")

    # Fine-Tune mô hình đã prune trên dữ liệu sạch
    print("Fine-tuning the pruned model on clean data...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(fine_tune_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader_clean, desc=f"Fine-Tuning Epoch {epoch+1}/{fine_tune_epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'Loss': f"{running_loss/total:.4f}", 'Acc': f"{100 * correct / total:.2f}%"})
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Fine-Tuning Epoch {epoch+1}/{fine_tune_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    # Đánh giá mô hình sau Fine-Tuning
    print("Evaluating fine-tuned model...")
    acc_clean_finetuned = evaluate_model(model, test_loader_clean, device=device)
    asr_finetuned = evaluate_asr(model, test_loader_bd, target_label=target_label, device=device)
    target_label_acc_finetuned = evaluate_target_label_accuracy(model, test_loader_clean, target_label=target_label, device=device)

    print(f"Fine-Tuned Model - Test Accuracy (Clean): {acc_clean_finetuned:.2f}%")
    print(f"Fine-Tuned Model - Attack Success Rate (ASR): {asr_finetuned:.2f}%")
    print(f"Fine-Tuned Model - Target Label Accuracy on Clean Data: {target_label_acc_finetuned:.2f}%")

    # Lưu mô hình đã fine-tune
    finetuned_model_path = os.path.join(save_path, 'model_fine_pruned_finetuned.pth')
    save_model(model, finetuned_model_path)
    print(f"Fine-tuned model saved at {finetuned_model_path}")

    # Trả về các chỉ số đánh giá
    metrics = {
        'pruned': {
            'acc_clean': acc_clean,
            'asr': asr,
            'target_label_acc': target_label_acc
        },
        'finetuned': {
            'acc_clean': acc_clean_finetuned,
            'asr': asr_finetuned,
            'target_label_acc': target_label_acc_finetuned
        }
    }

    return metrics
