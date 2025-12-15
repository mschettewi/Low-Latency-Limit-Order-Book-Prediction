from collections import Counter

import torch
from torch.utils.data import Dataset

OUT_PATH = "data/processed/fi2010_processed.pt"


class LobsterSequenceDataset(Dataset):
    def __init__(self, path: str, split: str = "train"):
        obj = torch.load(path, weights_only=False)
        self.lob = obj["lob"].float()
        self.seq_labels = obj["seq_labels"].long()
        self.window_size = int(obj["window_size"])
        self.horizon = int(obj["horizon"])

        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', 'test'")
        self.indices = obj[f"{split}_indices"].long()

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, idx):
        k = int(self.indices[idx])
        X = self.lob[k:k + self.window_size]
        y = self.seq_labels[k]
        return X, y


def analyze_class_distribution(dataset, name="Dataset"):
    """Check class distribution"""
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.item())

    counter = Counter(labels)
    total = len(labels)

    print(f"\n{'=' * 60}")
    print(f"{name} Class Distribution")
    print(f"{'=' * 60}")
    for cls in sorted(counter.keys()):
        count = counter[cls]
        pct = 100 * count / total
        print(f"Class {cls}: {count:,} samples ({pct:.2f}%)")
    print(f"Total: {total:,} samples")
    print(f"{'=' * 60}\n")

    return counter


def get_datasets(path=OUT_PATH):
    train_dataset = LobsterSequenceDataset(path, "train")
    val_dataset = LobsterSequenceDataset(path, "val")
    test_dataset = LobsterSequenceDataset(path, "test")

    return train_dataset, val_dataset, test_dataset


def analyze_data_distributions():
    train_dataset, val_dataset, test_dataset = get_datasets(OUT_PATH)
    print(f"Loaded: Train={len(train_dataset):,}, Val={len(val_dataset):,}, Test={len(test_dataset):,}")

    train_dist = analyze_class_distribution(train_dataset, "Training Set")
    val_dist = analyze_class_distribution(val_dataset, "Validation Set")
    test_dist = analyze_class_distribution(test_dataset, "Test Set")

    return train_dist, val_dist, test_dist


if __name__ == "__main__":
    analyze_data_distributions()
