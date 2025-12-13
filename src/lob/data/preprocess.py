import os

import numpy as np
import pandas as pd
import torch


def compute_normalization(orderbook: np.ndarray, train_frac_for_stats: float = 0.7):
    """Compute mean/std from training portion only"""
    N, F = orderbook.shape
    train_raw_end = int(train_frac_for_stats * N)
    train_slice = orderbook[:train_raw_end]

    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def make_window_labels_balanced(orderbook: np.ndarray, window_size: int, horizon: int, up_percentile: float = 67,
                                down_percentile: float = 33):
    """
    Create BALANCED labels using percentiles instead of fixed thresholds.
    This ensures roughly equal class distribution.

    Args:
        orderbook: Raw LOB data (N, F)
        window_size: Length of input sequence
        horizon: How far ahead to predict
        up_percentile: Percentile for up class (default 67 = top 33%)
        down_percentile: Percentile for down class (default 33 = bottom 33%)

    Returns:
        seq_labels: Array of class labels (0=down, 1=stationary, 2=up)
    """
    N, F = orderbook.shape

    # Compute mid-price
    best_ask = orderbook[:, 0]
    best_bid = orderbook[:, 2]
    mid = 0.5 * (best_ask + best_bid) / 10000.0

    M = N - window_size - horizon + 1
    if M <= 0:
        raise ValueError("window_size + horizon too large")

    t0 = np.arange(window_size - 1, window_size - 1 + M)
    t_future = t0 + horizon
    diff = mid[t_future] - mid[t0]

    # Use percentiles to create balanced classes
    up_threshold = np.percentile(diff, up_percentile)
    down_threshold = np.percentile(diff, down_percentile)

    seq_labels = np.ones(M, dtype=np.int64)  # Default: stationary (1)
    seq_labels[diff > up_threshold] = 2  # Up
    seq_labels[diff < down_threshold] = 0  # Down

    # Print distribution
    unique, counts = np.unique(seq_labels, return_counts=True)
    print(f"\n{'=' * 60}")
    print("BALANCED LABEL DISTRIBUTION")
    print(f"{'=' * 60}")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c:,} ({100 * c / M:.1f}%)")
    print(f"  Up threshold: {up_threshold:.6f}")
    print(f"  Down threshold: {down_threshold:.6f}")
    print(f"{'=' * 60}\n")

    return seq_labels


def build_split_indices(num_windows: int, train_frac: float = 0.7, val_frac: float = 0.15, mode: str = "chronological",
                        seed: int = 42):
    """Split dataset into train/val/test"""
    M = num_windows
    train_end = int(train_frac * M)
    val_end = int((train_frac + val_frac) * M)

    if mode == "chronological":
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(train_end, val_end, dtype=np.int64)
        test_idx = np.arange(val_end, M, dtype=np.int64)
    elif mode == "random":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(M)
        train_idx = perm[:train_end]
        val_idx = perm[train_end:val_end]
        test_idx = perm[val_end:]
    else:
        raise ValueError("mode must be 'chronological' or 'random'")

    return train_idx, val_idx, test_idx


def process_data():
    # Process data with balanced labels
    window_size = 100
    horizon = 10

    files_in_raw = os.listdir("data/raw")
    RAW_ORDERBOOK_NAME = [f for f in files_in_raw if "orderbook" in f][0]
    RAW_MESSAGE_NAME = [f for f in files_in_raw if "message" in f][0]

    RAW_ORDERBOOK_PATH = f"data/raw/{RAW_ORDERBOOK_NAME}"
    OUT_PATH = "data/processed/aapl_lobster_balanced.pt"

    print(f"Loading orderbook from: {RAW_ORDERBOOK_PATH}")
    ob = pd.read_csv(RAW_ORDERBOOK_PATH, header=None).values.astype(np.float32)
    N, F = ob.shape
    print(f"Orderbook shape: N={N}, F={F}")

    # Normalize
    mean, std = compute_normalization(ob, train_frac_for_stats=0.7)
    ob_norm = (ob - mean) / std

    # Create BALANCED labels using percentiles
    seq_labels = make_window_labels_balanced(orderbook=ob, window_size=window_size, horizon=horizon, up_percentile=67,
                                             # Top 33% = up
                                             down_percentile=33,  # Bottom 33% = down
                                             )
    M = seq_labels.shape[0]
    print(f"Number of valid windows (M) = {M}")

    # Split data
    train_idx, val_idx, test_idx = build_split_indices(num_windows=M, train_frac=0.7, val_frac=0.15,
                                                       mode="chronological",
                                                       seed=42, )
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save processed data
    obj = {"lob": torch.from_numpy(ob_norm), "seq_labels": torch.from_numpy(seq_labels), "window_size": window_size,
           "horizon": horizon, "train_indices": torch.from_numpy(train_idx), "val_indices": torch.from_numpy(val_idx),
           "test_indices": torch.from_numpy(test_idx), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std), }

    torch.save(obj, OUT_PATH)
    print(f"Saved processed dataset to: {OUT_PATH}")


if __name__ == "__main__":
    process_data()
