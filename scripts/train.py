import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lob.data.dataset import get_datasets
from src.lob.models.transformer import LOBTransformer, ModelConfig
from src.lob.models.cnn import LOBCNN
from src.lob.train.train import train_model


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


def train_transformer_base(device, train_dataset, val_dataset, test_dataset):
    # Compute class weights from training data
    class_weights = compute_class_weights(train_dataset, device)

    # run training

    # Configure model
    config = ModelConfig.base()
    config.num_epochs = 30
    config.use_amp = True
    config.use_compile = False  # Set True if using GPU with torch 2.0+

    config.print_summary("Model Configuration")

    # Create model
    model = LOBTransformer(config)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Train with balanced data + weighted loss
    metrics = train_model(
        model,
        train_dataset,
        val_dataset,
        config,
        device,
        class_weights,
        "transformer_balanced.pt",
        wandb_name='transformer_balanced',
        wandb_group='Transformer',
        wandb_config={
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'use_amp': config.use_amp,
            'use_compile': config.use_compile,
            'seq_length': config.seq_length,
            'input_dim': config.input_dim,
            'num_classes': config.num_classes,
        }
    )


def train_cnn_base(device, train_dataset, val_dataset, test_dataset):
    print("Using device:", device)

    # Infer shapes from the dataset correctly
    # lob: [num_events, num_features]
    input_dim = train_dataset.lob.shape[1]  # F (40)
    seq_length = train_dataset.window_size  # T (sequence length)
    num_classes = int(train_dataset.seq_labels.max().item() + 1)

    print("seq_length:", seq_length)
    print("input_dim:", input_dim)
    print("num_classes:", num_classes)

    # Start from the existing ModelConfig (or ModelConfig.base() if that's what you use)
    cnn_config = ModelConfig()  # or ModelConfig.base()

    # Overwrite the fields that matter for CNN
    cnn_config.seq_length = seq_length
    cnn_config.input_dim = input_dim
    cnn_config.num_classes = num_classes

    # You can tweak training hyperparams here if you want
    cnn_config.num_epochs = 20  # e.g. fewer epochs for a first run
    cnn_config.batch_size = 256
    cnn_config.learning_rate = 1e-3
    cnn_config.use_amp = True
    cnn_config.use_compile = False

    # Instantiate the CNN model
    cnn_model = LOBCNN(cnn_config).to(device)

    # Compute class weights from the training dataset (reuse existing helper)
    class_weights = compute_class_weights(train_dataset, device)

    # Train the CNN using the same training harness as the Transformer
    cnn_ckpt_name = "cnn_baseline.pt"

    cnn_metrics = train_model(
        model=cnn_model,
        train_ds=train_dataset,
        val_ds=val_dataset,
        config=cnn_config,
        device=device,
        class_weights=class_weights,
        ckpt_name=cnn_ckpt_name,
        wandb_name='cnn_baseline',
        wandb_group='CNN',
        wandb_config={
            'learning_rate': cnn_config.learning_rate,
            'batch_size': cnn_config.batch_size,
            'num_epochs': cnn_config.num_epochs,
            'use_amp': cnn_config.use_amp,
            'use_compile': cnn_config.use_compile,
            'seq_length': cnn_config.seq_length,
            'input_dim': cnn_config.input_dim,
            'num_classes': cnn_config.num_classes,
        }
    )

    print("Finished training CNN baseline.")


def train_transformer_small(device, train_dataset, val_dataset, test_dataset):
    # Train small model
    config_small = ModelConfig.small()
    config_small.num_epochs = 20
    config_small.use_amp = True

    config_small.print_summary("Small Model Configuration")

    model_small = LOBTransformer(config_small)
    model = LOBTransformer(ModelConfig.base())
    print(f"Small model parameters: {model_small.count_parameters():,}")
    print(f"Base model parameters: {model.count_parameters():,}")
    print(f"Parameter reduction: {100 * (1 - model_small.count_parameters() / model.count_parameters()):.1f}%")

    class_weights = compute_class_weights(train_dataset, device)

    # Train small model
    metrics_small = train_model(
        model_small,
        train_dataset,
        val_dataset,
        config_small,
        device,
        class_weights,
        "transformer_small.pt",
        wandb_name='transformer_small',
        wandb_group='Transformer',
        wandb_config={
            'learning_rate': config_small.learning_rate,
            'batch_size': config_small.batch_size,
            'num_epochs': config_small.num_epochs,
            'use_amp': config_small.use_amp,
            'use_compile': config_small.use_compile,
            'seq_length': config_small.seq_length,
            'input_dim': config_small.input_dim,
            'num_classes': config_small.num_classes,
        }
    )

    print("Finished training Transformer small.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_transformer_base(device, train_dataset, val_dataset, test_dataset)
    train_cnn_base(device, train_dataset, val_dataset, test_dataset)

    train_transformer_small(device, train_dataset, val_dataset, test_dataset)
