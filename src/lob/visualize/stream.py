import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time


@torch.no_grad()
def visualize(
        model,
        obj,
        samples=2000,
        visualization_sleep=0.001
):
    """Stream LOB mid-prices with model predictions.

    The model consumes a sliding window of length ``window_size`` and predicts the
    class of the mid-price movement ``horizon`` steps *after* the end of the
    window (e.g., with ``horizon=10`` we predict the direction 10 ticks in the
    future).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = model.to(device).eval()

    lob = obj["lob"]
    N, F = lob.shape
    seq_labels = obj["seq_labels"]
    window_size = obj["window_size"]
    horizon = obj["horizon"]
    mean = obj["mean"]
    std = obj["std"]

    # lob_not_normalize = lob * std + mean

    # Compute mid-price
    best_ask_raw = lob[:, 0]
    best_bid_raw = lob[:, 2]
    mid = 0.5 * (best_ask_raw + best_bid_raw) / 10000.0

    # Valid number of windows that have labels (accounts for horizon)
    M = seq_labels.shape[0]
    t0 = np.arange(window_size - 1, window_size - 1 + M)
    t_future = t0 + horizon  # index of the price we forecast

    preds = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    plot_limit = min(samples, len(mid))
    ax1.plot(np.arange(plot_limit), mid[:plot_limit])
    ax1.set_title(
        f"Mid-price with sliding window (predicting {horizon} ticks after window end)")
    ax1.set_ylabel("Mid-price")

    shaded_x0 = 0
    shade = ax1.axvspan(shaded_x0, shaded_x0 + window_size - 1, color='gray', alpha=0.1)
    target_line = ax1.axvline(t_future[0], color='tab:red', linestyle='--', alpha=0.6)
    target_dot = ax1.plot(t_future[0], mid[t_future[0]], marker='o', color='tab:red')[0]

    display(fig)

    class_names = {0: "down", 1: "flat", 2: "up"}

    for k in range(M):
        window_start = k
        window_end = k + window_size
        target_time = t_future[k]

        # Stop early if we would go past the available prices when plotting
        if target_time >= len(mid):
            break

        X = lob[window_start: window_end]
        logits = model(X.unsqueeze(0).to(device))
        logits = torch.softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1).item()
        preds.append(pred)

        shade.remove()
        shade = ax1.axvspan(window_start, window_end - 1, color='gray', alpha=0.1)

        target_line.remove()
        target_dot.remove()
        target_line = ax1.axvline(target_time, color='tab:red', linestyle='--', alpha=0.6)
        target_dot = ax1.plot(target_time, mid[target_time], marker='o', color='tab:red')[0]

        w = min(50, len(preds))

        ax2.clear()
        xs = np.arange(window_start, window_end)
        ys = mid[window_start:window_end]
        ax2.plot(xs, ys, label="Window mid-price")
        target_price = mid[target_time]
        ax2.scatter(target_time, target_price, color='tab:red', label=f"Target (t+{horizon})")
        ax2.annotate(
            f"t+{horizon} price = {target_price:.4f}",
            (target_time, target_price),
            textcoords="offset points",
            xytext=(10, 8),
            color="tab:red",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:red", alpha=0.7),
        )
        xlim_left = max(0, window_start - 2)
        xlim_right = target_time + 2
        ax2.set_xlim(xlim_left, xlim_right)
        ax2.text(
            0.01,
            0.97,
            f"Predicted class: {class_names.get(pred, pred)}\nActual class: {class_names.get(seq_labels[k], seq_labels[k])}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax2.set_xlabel("LOB index (window and target positions)")
        ax2.legend()
        ax2.set_ylabel("Mid-price")

        ax3.clear()
        idx_slice = slice(len(preds) - w, len(preds))
        time_slice = t_future[idx_slice]
        ax3.plot(time_slice, seq_labels[idx_slice], label="Actual", drawstyle="steps-mid")
        ax3.plot(time_slice, preds[-w:], label="Predicted", drawstyle="steps-mid")
        ax3.set_ylabel("Class (0=down, 1=flat, 2=up)")
        ax3.set_xlabel("LOB index of target price (window end + horizon)")
        ax3.set_xlim(xlim_left, xlim_right)
        ax3.legend()

        clear_output(wait=True)
        display(fig)

        time.sleep(visualization_sleep)

