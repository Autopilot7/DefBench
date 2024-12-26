# defenses/activation_clustering.py

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def activation_clustering_defense(model, train_loader, target_label, device='cpu', layer_name='fc1'):
    """
    Áp dụng phương pháp Activation Clustering để phát hiện và loại bỏ các mẫu bị nhiễm.

    Args:
        model (torch.nn.Module): Mô hình đã được đào tạo.
        train_loader (DataLoader): DataLoader chứa dữ liệu đào tạo bị nhiễm.
        target_label (int): Nhãn mục tiêu của backdoor.
        device (str): Thiết bị ('cpu' hoặc 'cuda').
        layer_name (str): Tên của lớp để trích xuất kích hoạt.

    Returns:
        DataLoader: DataLoader đã được làm sạch (loại bỏ các mẫu bị nhiễm).
    """
    activations = []
    labels_list = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Đăng ký hook
    hook = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break

    if hook is None:
        raise ValueError(f"Lớp {layer_name} không tồn tại trong mô hình.")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(train_loader, desc="Extracting activations"):
            inputs = inputs.to(device)
            _ = model(inputs)
            labels_list.extend(labels.numpy())

    # Gỡ bỏ hook
    hook.remove()

    # Chuyển đổi danh sách thành mảng numpy
    activations = np.concatenate(activations, axis=0)
    labels = np.array(labels_list)

    # Giảm chiều dữ liệu bằng PCA
    pca = PCA(n_components=50)
    reduced_activations = pca.fit_transform(activations)

    # Áp dụng K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_activations)

    # Xác định nhóm bị nhiễm
    poisoned_cluster = None
    for cluster in np.unique(cluster_labels):
        cluster_true_labels = labels[cluster_labels == cluster]
        target_ratio = np.sum(cluster_true_labels == target_label) / len(cluster_true_labels)
        if target_ratio > 0.5:  # Ngưỡng có thể điều chỉnh
            poisoned_cluster = cluster
            break

    if poisoned_cluster is not None:
        poisoned_indices = np.where(cluster_labels == poisoned_cluster)[0]
        print(f"Đã phát hiện {len(poisoned_indices)} mẫu bị nhiễm.")
    else:
        poisoned_indices = []
        print("Không phát hiện mẫu bị nhiễm nào.")

    # Tạo bộ dữ liệu sạch
    clean_indices = list(set(range(len(labels))) - set(poisoned_indices))
    clean_images = []
    clean_labels = []

    for idx, (images, label) in enumerate(train_loader):
        if idx * train_loader.batch_size >= len(labels):
            break
        for i in range(images.size(0)):
            global_idx = idx * train_loader.batch_size + i
            if global_idx in clean_indices:
                clean_images.append(images[i].unsqueeze(0))
                clean_labels.append(label[i].unsqueeze(0))

    if len(clean_images) == 0:
        raise ValueError("Không tìm thấy mẫu dữ liệu sạch sau khi loại bỏ các mẫu bị nhiễm.")

    clean_images = torch.cat(clean_images, dim=0)
    clean_labels = torch.cat(clean_labels, dim=0)
    clean_dataset = TensorDataset(clean_images, clean_labels)
    clean_loader = DataLoader(clean_dataset, batch_size=train_loader.batch_size, shuffle=True)

    print(f"Số lượng mẫu dữ liệu sạch: {len(clean_dataset)}")
    return clean_loader
