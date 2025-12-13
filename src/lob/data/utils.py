from collections import Counter
import torch

def compute_class_weights(dataset, device):
    """Compute inverse frequency weights for balanced training"""
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.item())

    counter = Counter(labels)
    total = len(labels)

    # Inverse frequency weighting
    weights = torch.zeros(3)
    for cls in range(3):
        if cls in counter:
            weights[cls] = total / (3 * counter[cls])
        else:
            weights[cls] = 1.0

    weights = weights.to(device)
    print(f"\nClass weights (inverse frequency): {weights.cpu().numpy()}")
    return weights

# Compute class weights from training data
class_weights = compute_class_weights(train_dataset, device)