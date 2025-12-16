import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.lob.models.transformer import LOBTransformer, ModelConfig
from src.lob.evals.quantized_model import benchmark_quantization, get_model_accuracy
from src.lob.data.dataset import get_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    transformer_config = ModelConfig.base()
    transformer_config.num_epochs = 30
    transformer_config.use_amp = True
    transformer_config.use_compile = False  # Set True if using GPU with torch 2.0+
    transformer_config.print_summary("Model Configuration")

    model = LOBTransformer(transformer_config).to(device)

    quantized_model = benchmark_quantization(model, transformer_config)

    # Load Test Dataset
    _, _, test_dataset = get_datasets()

    # 1. Get Baseline Accuracy (FP32)
    # Ensure test_dataset is loaded from your previous cells
    print("\nMeasuring Accuracy Impact...")
    acc_fp32 = get_model_accuracy(model, test_dataset)
    print(f"Original Model (FP32) Accuracy: {acc_fp32:.2%}")

    # 2. Get Quantized Accuracy (Int8)
    acc_int8 = get_model_accuracy(quantized_model, test_dataset)
    print(f"Quantized Model (Int8) Accuracy: {acc_int8:.2%}")

    # 3. Report the Drop
    print(f"Accuracy Drop: {acc_fp32 - acc_int8:.2%}")
