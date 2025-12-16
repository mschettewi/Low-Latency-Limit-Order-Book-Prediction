import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader


def evaluate_model(model, test_ds, config, device, name=""):
    """Comprehensive model evaluation"""
    print(f"\n===== Evaluating {name} =====")
    model.to(device)
    model.eval()

    loader = DataLoader(test_ds, config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_preds, all_labels = [], []
    total_loss, total = 0, 0

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            preds = torch.argmax(logits, 1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_loss = total_loss / total
    test_acc = (all_preds == all_labels).float().mean().item()

    print(f"\n{'=' * 70}")
    print(f"Test Results")
    print(f"{'=' * 70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Per-class accuracy
    class_names = ['Down (0)', 'Stationary (1)', 'Up (2)']
    print(f"\nPer-class Accuracy:")
    for c in range(3):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
            count = mask.sum().item()
            print(f"  {class_names[c]}: {acc:.4f} ({count:,} samples)")

    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())

    print(f"\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(3)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(3):
        for j in range(3):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'logs/confusion_matrix_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Classification report
    print(f"\nClassification Report:")
    classification_report_str = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=class_names, digits=4)
    print(classification_report_str)

    with open(f'logs/classification_report_{name}.txt', 'w') as f:
        f.write(classification_report_str)

    results = {'test_loss': test_loss, 'test_acc': test_acc, 'predictions': all_preds, 'labels': all_labels,
            'confusion_matrix': cm}

    # Adjust keys if your evaluate_model dict uses different names
    test_acc = results.get("test_acc", None)
    test_loss = results.get("test_loss", None)

    if test_loss is not None:
        print(f"{name} test loss: {test_loss:.4f}")
    if test_acc is not None:
        print(f"{name} test accuracy: {test_acc:.4f}")

    return results
