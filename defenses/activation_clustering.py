# defenses/activation_clustering.py

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract_activations(model, dataloader, device='cpu', layer='fc1'):
    """
    Trích xuất các kích hoạt từ một lớp nhất định trong mô hình.
    
    Args:
        model (torch.nn.Module): Mô hình đã được đào tạo.
        dataloader (DataLoader): DataLoader chứa dữ liệu để trích xuất kích hoạt.
        device (str): Thiết bị ('cpu' hoặc 'cuda').
        layer (str): Tên của lớp để trích xuất kích hoạt.
    
    Returns:
        np.ndarray: Mảng các kích hoạt.
        np.ndarray: Mảng các nhãn tương ứng.
    """
    activations = []
    labels_list = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Đăng ký hook
    hook = None
    for name, module in model.named_modules():
        if name == layer:
            hook = module.register_forward_hook(hook_fn)
            break
    
    if hook is None:
        raise ValueError(f"Lớp {layer} không tồn tại trong mô hình.")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting activations"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels_list.extend(labels.numpy())

    # Gỡ bỏ hook
    hook.remove()

    # Chuyển đổi danh sách thành mảng numpy
    activations = np.concatenate(activations, axis=0)
    labels = np.array(labels_list)

    return activations, labels

def perform_clustering(activations, n_clusters=2):
    """
    Thực hiện phân nhóm K-Means trên các kích hoạt.
    
    Args:
        activations (np.ndarray): Mảng các kích hoạt.
        n_clusters (int): Số lượng nhóm.
    
    Returns:
        np.ndarray: Nhãn phân nhóm.
    """
    # Giảm chiều dữ liệu bằng PCA để tăng tốc độ và hiệu quả
    pca = PCA(n_components=50)
    reduced_activations = pca.fit_transform(activations)

    # Áp dụng K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_activations)

    return cluster_labels

def identify_poisoned_clusters(cluster_labels, true_labels, target_label):
    """
    Xác định nhóm chứa dữ liệu bị nhiễm dựa trên nhãn mục tiêu.
    
    Args:
        cluster_labels (np.ndarray): Nhãn phân nhóm.
        true_labels (np.ndarray): Nhãn thực tế của các mẫu.
        target_label (int): Nhãn mục tiêu của backdoor.
    
    Returns:
        list: Các chỉ số của các mẫu bị nghi ngờ bị nhiễm.
    """
    clusters = np.unique(cluster_labels)
    poisoned_cluster = None
    cluster_sizes = {}

    for cluster in clusters:
        cluster_size = np.sum(cluster_labels == cluster)
        cluster_sizes[cluster] = cluster_size

    # Giả định rằng nhóm có tỷ lệ mẫu nhãn mục tiêu cao hơn là nhóm bị nhiễm
    for cluster in clusters:
        cluster_true_labels = true_labels[cluster_labels == cluster]
        target_ratio = np.sum(cluster_true_labels == target_label) / len(cluster_true_labels)
        if target_ratio > 0.2:  # Ngưỡng có thể điều chỉnh
            poisoned_cluster = cluster
            break

    if poisoned_cluster is not None:
        poisoned_indices = np.where(cluster_labels == poisoned_cluster)[0]
        return poisoned_indices.tolist()
    else:
        return []

def activation_clustering_defense(model, train_loader, target_label, device='cpu', layer='fc1'):
    """
    Áp dụng phương pháp Activation Clustering để phát hiện và loại bỏ các mẫu bị nhiễm.
    
    Args:
        model (torch.nn.Module): Mô hình đã được đào tạo.
        train_loader (DataLoader): DataLoader chứa dữ liệu đào tạo.
        target_label (int): Nhãn mục tiêu của backdoor.
        device (str): Thiết bị ('cpu' hoặc 'cuda').
        layer (str): Tên của lớp để trích xuất kích hoạt.
    
    Returns:
        DataLoader: DataLoader đã được làm sạch (loại bỏ các mẫu bị nhiễm).
    """
    # 1. Trích xuất kích hoạt từ lớp cụ thể
    activations, labels = extract_activations(model, train_loader, device=device, layer=layer)

    # 2. Thực hiện phân nhóm K-Means
    cluster_labels = perform_clustering(activations)

    # 3. Xác định các nhóm bị nhiễm
    poisoned_indices = identify_poisoned_clusters(cluster_labels, labels, target_label)

    print(f"Đã phát hiện {len(poisoned_indices)} mẫu bị nhiễm.")

    # 4. Tạo bộ dữ liệu sạch
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
    clean_dataset = torch.utils.data.TensorDataset(clean_images, clean_labels)
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=train_loader.batch_size, shuffle=True)

    print(f"Số lượng mẫu dữ liệu sạch: {len(clean_dataset)}")
    return clean_loader
