import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lob.data.dataset import get_datasets
from src.lob.evals.evaluate import evaluate_model
from src.lob.evals.benchmark import benchmark_model
from src.lob.models.config import ModelConfig
from src.lob.models.transformer import LOBTransformer
from src.lob.models.cnn import LOBCNN

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_project = "HPML LOB"

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_datasets()

    ###### Transformer ########
    # 1) Transformer ('transformer_balanced.pt' and 'config')
    transformer_config = ModelConfig.base()
    transformer_config.num_epochs = 30
    transformer_config.use_amp = True
    transformer_config.use_compile = False  # Set True if using GPU with torch 2.0+
    transformer_config.print_summary("Model Configuration")

    model = LOBTransformer(transformer_config).to(device)

    evaluate_model(model=model, test_ds=test_dataset, config=transformer_config, device=device,
                   name="transformer_balanced")
    benchmark_model(model=model, config=transformer_config, device=device, wandb_group="Transformer",
                    wandb_name="transformer_balanced")

    ####### CNN ########
    # 2) CNN baseline
    cnn_config = ModelConfig()

    input_dim = train_dataset.lob.shape[1]  # F (40)
    seq_length = train_dataset.window_size  # T (sequence length)
    num_classes = int(train_dataset.seq_labels.max().item() + 1)

    # Overwrite the fields that matter for CNN
    cnn_config.seq_length = seq_length
    cnn_config.input_dim = input_dim
    cnn_config.num_classes = num_classes

    cnn_config.num_epochs = 20  # e.g. fewer epochs for a first run
    cnn_config.batch_size = 256
    cnn_config.learning_rate = 1e-3
    cnn_config.use_amp = True
    cnn_config.use_compile = False

    model = LOBCNN(cnn_config).to(device)

    evaluate_model(model=model, test_ds=test_dataset, config=cnn_config, device=device, name="cnn_baseline")
    benchmark_model(model=model, config=cnn_config, device=device, wandb_group="CNN", wandb_name="cnn_baseline")

    ######## Transformer Small ########
    config_small = ModelConfig.small()
    config_small.num_epochs = 20
    config_small.use_amp = True

    config_small.print_summary("Small Model Configuration")

    model_small = LOBTransformer(config_small).to(device)

    evaluate_model(model=model_small, test_ds=test_dataset, config=config_small, device=device,
                   name="transformer_small")
    benchmark_model(model=model_small, config=config_small, device=device, wandb_group="Transformer",
                    wandb_name="transformer_small")
