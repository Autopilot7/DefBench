# defenses/strip.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import csv
import os

from metrics.evaluation import evaluate_model, evaluate_asr, evaluate_target_label_accuracy
from utils.helpers import save_model, load_model, train_model, create_poisoned_loader, get_device, set_seed

def strip_defense(model, clean_loader, poisoned_loader, device='cpu', strip_alpha=1.0, N=64, defense_fpr=0.05, save_path='./models'):
    """
    Áp dụng phương pháp STRIP để phát hiện các mẫu bị nhiễm trong tập kiểm tra.

    Args:
        model (torch.nn.Module): Mô hình đã được đào tạo.
        clean_loader (DataLoader): DataLoader chứa dữ liệu sạch.
        poisoned_loader (DataLoader): DataLoader chứa dữ liệu bị nhiễm.
        device (str): Thiết bị để chạy STRIP ('cpu' hoặc 'cuda').
        strip_alpha (float): Hệ số pha trộn ảnh.
        N (int): Số lượng ảnh sạch để pha trộn với mỗi ảnh kiểm tra.
        defense_fpr (float): Ngưỡng False Positive Rate để xác định ngưỡng entropy.
        save_path (str): Đường dẫn để lưu kết quả.

    Returns:
        dict: Các chỉ số đánh giá sau STRIP.
    """
    model.to(device)
    model.eval()

    # Tạo thư mục lưu kết quả nếu chưa tồn tại
    os.makedirs(save_path, exist_ok=True)
    detection_info_path = os.path.join(save_path, 'strip_detection_info.csv')

    # Hàm tính entropy
    def entropy(output):
        p = nn.Softmax(dim=1)(output) + 1e-8
        return (-p * torch.log(p)).sum(1)

    # Bước 1: Thu thập entropy cho dữ liệu sạch để xác định ngưỡng
    print("Collecting entropy from clean samples to determine thresholds...")
    clean_entropies = []

    for inputs, _ in tqdm(clean_loader, desc="Processing clean data"):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            ent = entropy(outputs).cpu().numpy()
            clean_entropies.extend(ent)

    clean_entropies = np.array(clean_entropies)
    clean_entropies_sorted = np.sort(clean_entropies)
    threshold_low = clean_entropies_sorted[int(defense_fpr * len(clean_entropies_sorted))]
    threshold_high = np.inf  # STRIP thường chỉ sử dụng ngưỡng thấp

    print(f"Threshold Low (based on defense FPR={defense_fpr}): {threshold_low}")

    # Bước 2: Áp dụng STRIP cho tập kiểm tra (cả sạch và bị nhiễm)
    print("Applying STRIP to the inspection set (clean and poisoned data)...")
    inspection_entropies = []
    true_labels = []
    suspected = []

    # Kết hợp cả clean và poisoned data
    combined_inputs = torch.cat([inputs for inputs, _ in clean_loader], dim=0) if len(clean_loader.dataset) > 0 else torch.tensor([])
    combined_inputs = torch.cat([combined_inputs, torch.cat([inputs for inputs, _ in poisoned_loader], dim=0)], dim=0)
    combined_labels = np.concatenate([labels.numpy() for _, labels in clean_loader] + [labels.numpy() for _, labels in poisoned_loader], axis=0)

    combined_dataset = TensorDataset(combined_inputs, torch.tensor(combined_labels))
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)

    for inputs, labels in tqdm(combined_loader, desc="Processing inspection data"):
        inputs = inputs.to(device)
        batch_size = inputs.size(0)
        for i in range(batch_size):
            test_input = inputs[i].unsqueeze(0)  # Shape: [1, C, H, W]
            # Chọn N ảnh sạch ngẫu nhiên để pha trộn
            if len(clean_loader.dataset) > 0:
                clean_indices = random.sample(range(len(clean_loader.dataset)), min(N, len(clean_loader.dataset)))
                clean_samples = torch.stack([clean_loader.dataset[i][0] for i in clean_indices]).to(device)
                # Pha trộn test_input với các clean_samples
                mixed_inputs = test_input.repeat(N, 1, 1, 1) + strip_alpha * clean_samples
                with torch.no_grad():
                    outputs = model(mixed_inputs)
                    entropies = entropy(outputs).cpu().numpy()
                    avg_entropy = entropies.mean()
            else:
                # Nếu không có dữ liệu sạch để pha trộn, sử dụng entropy trực tiếp
                with torch.no_grad():
                    outputs = model(test_input)
                    avg_entropy = entropy(outputs).cpu().numpy()[0]

            inspection_entropies.append(avg_entropy)
            true_labels.append(labels[i].item())
            # Đánh dấu nếu entropy < threshold_low hoặc > threshold_high
            if avg_entropy < threshold_low or avg_entropy > threshold_high:
                suspected.append(1)  # 1: Bị nghi ngờ là bị nhiễm
            else:
                suspected.append(0)  # 0: Sạch

    inspection_entropies = np.array(inspection_entropies)
    true_labels = np.array(true_labels)
    suspected = np.array(suspected)

    # Bước 3: Tính toán các chỉ số đánh giá
    print("Calculating evaluation metrics...")
    # Tạo nhãn thực tế: 1 nếu bị nhiễm, 0 nếu sạch
    num_clean = len(clean_loader.dataset)
    num_poisoned = len(poisoned_loader.dataset)
    true_poisoned = np.concatenate([np.zeros(num_clean), np.ones(num_poisoned)])

    # Tính confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_poisoned, suspected).ravel()

    # Tính các chỉ số
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0

    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"TPR (Recall): {TPR:.4f}")
    print(f"FPR: {FPR:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Lưu kết quả vào CSV
    with open(detection_info_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['TP', 'TN', 'FP', 'FN', 'TPR', 'FPR', 'Precision', 'Accuracy', 'F1_Score'])
        csvwriter.writerow([tp, tn, fp, fn, TPR, FPR, precision, acc, f1])

    metrics = {
        'confusion_matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
        'metrics': {
            'TPR': TPR,
            'FPR': FPR,
            'Precision': precision,
            'Accuracy': acc,
            'F1_Score': f1
        },
        'detection_info_path': detection_info_path
    }

    print(f"Detection results saved at {detection_info_path}")

    return metrics

