# attacks/badnets.py
import torch
import numpy as np

def add_trigger(images, trigger_pattern, target_label, trigger_position=(0,0)):
    triggered_images = images.clone()
    _, _, h, w = images.shape
    th, tw = trigger_pattern.shape
    x, y = trigger_position
    triggered_images[:, :, x:x+th, y:y+tw] = torch.from_numpy(trigger_pattern).float()
    labels = torch.full((images.size(0),), target_label, dtype=torch.long)
    return triggered_images, labels

def apply_badnets(train_loader, trigger_pattern, target_label, poison_rate=0.1):
    poisoned_data = []
    for images, labels in train_loader:
        if np.random.rand() < poison_rate:
            images, poisoned_labels = add_trigger(images, trigger_pattern, target_label)
            poisoned_data.append((images, poisoned_labels))
        else:
            poisoned_data.append((images, labels))
    return poisoned_data
