import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lob.data.dataset import get_datasets
from src.lob.evals.evaluate import evaluate_model
from src.lob.models.cnn import LOBCNN
from src.lob.models.config import ModelConfig
from src.lob.models.transformer import LOBTransformer

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset = get_datasets()


def evaluate_and_print(name, model_cls, config, ckpt_name):
    print(f"\n===== Evaluating {name} =====")
    model = model_cls(config).to(device)
    ckpt_path = f"checkpoints/{ckpt_name}"

    results = evaluate_model(model=model, test_ds=test_dataset, config=config, device=device, ckpt_path=ckpt_path,
                             name=name)

    # Adjust keys if your evaluate_model dict uses different names
    test_acc = results.get("test_acc", None)
    test_loss = results.get("test_loss", None)

    if test_loss is not None:
        print(f"{name} test loss: {test_loss:.4f}")
    if test_acc is not None:
        print(f"{name} test accuracy: {test_acc:.4f}")

    return results


transformer_config = ModelConfig.base()
transformer_config.num_epochs = 30
transformer_config.use_amp = True
transformer_config.use_compile = False  # Set True if using GPU with torch 2.0+

transformer_config.print_summary("Model Configuration")

# 1) Transformer ('transformer_balanced.pt' and 'config')
transformer_results = evaluate_and_print(name="Transformer", model_cls=LOBTransformer, config=transformer_config,
                                         ckpt_name="transformer_balanced.pt", )

cnn_config = ModelConfig()  # or ModelConfig.base()

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

# 2) CNN baseline
cnn_results = evaluate_and_print(name="CNN baseline", model_cls=LOBCNN, config=cnn_config,
                                 ckpt_name="cnn_baseline.pt", )
