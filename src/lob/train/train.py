import time

import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .metrics import MetricsTracker
from .scheduler import get_lr_scheduler

wandb_project = "HPML LOB"


def train_epoch(model, loader, optimizer, scheduler, scaler, config, device, class_weights):
    """Training epoch with weighted loss"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if config.use_amp:
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += loss.item() * x.size(0)
        correct += (torch.argmax(logits, 1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, config, device, class_weights):
    """Validation with weighted loss"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if config.use_amp:
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=class_weights)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)

        total_loss += loss.item() * x.size(0)
        correct += (torch.argmax(logits, 1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def train_model(model, train_ds, val_ds, config, device, class_weights, ckpt_name="best.pt", wandb_group="",
                wandb_name="", wandb_config={}):
    """Complete training loop with class-weighted loss"""
    train_loader = DataLoader(
        train_ds, config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = model.to(device)

    if config.use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled!")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=config.use_amp)

    tracker = MetricsTracker(log_wandb=True)
    best_val_acc = 0.0

    print(f"\n{'=' * 70}")
    print(f"Training {config.num_epochs} epochs with BALANCED DATA + WEIGHTED LOSS")
    print(f"{'=' * 70}")
    print(f"AMP={config.use_amp}, compile={config.use_compile}")
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")
    print(f"Class weights: {class_weights.cpu().numpy()}\n")

    with wandb.init(project=wandb_project, config=config, group=wandb_group, name=wandb_name) as run:
        for epoch in range(config.num_epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, config, device, class_weights
            )
            val_loss, val_acc = validate(
                model, val_loader, config, device, class_weights
            )
            lr = optimizer.param_groups[0]['lr']

            tracker.update(train_loss, train_acc, val_loss, val_acc, lr, epoch)

            marker = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config,
                    'class_weights': class_weights
                }, f"checkpoints/{ckpt_name}")
                marker = " [BEST]"

            print(f"Epoch {epoch + 1:3d}/{config.num_epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                  f"LR: {lr:.6f} | {time.time() - t0:.1f}s{marker}")

        print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")
        tracker.plot(wandb_name)
        return tracker
