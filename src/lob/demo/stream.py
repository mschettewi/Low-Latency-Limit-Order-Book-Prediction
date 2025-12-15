import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


@torch.no_grad()
def visualize(
        model,
        obj,
        samples=2000,
        visualization_sleep=0.001
):
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

    lob_not_normalize = lob * std + mean

    # Compute mid-price
    best_ask_raw = lob_not_normalize[:, 0]
    best_bid_raw = lob_not_normalize[:, 2]
    mid = 0.5 * (best_ask_raw + best_bid_raw) / 10000.0

    preds = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    ax1.plot(mid[:samples])

    shaded_x0 = 0
    shade = ax1.axvspan(shaded_x0, shaded_x0 + window_size - 1, color='gray', alpha=0.1)

    fig.show()

    for k in range(0, N - window_size + 1):
        X = lob[k: k + window_size]
        logits = model(X.unsqueeze(0).to(device))
        logits = torch.softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1).item()
        preds.append(pred)

        shade.remove()
        shade = ax1.axvspan(k, k + window_size - 1, color='gray', alpha=0.1)

        w = min(50, len(preds))

        ax2.clear()
        xs = np.arange(k, k + window_size)
        ys = mid[k:k + window_size]
        ax2.plot(xs, ys)

        ax3.clear()
        ax3.plot(seq_labels[len(preds) - w:len(preds)], label="Actual")
        ax3.plot(preds[-w:], label="Predicted")
        ax3.legend()

        clear_output(wait=True)
        fig.show()

        time.sleep(visualization_sleep)


