import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.lob.models.transformer import LOBTransformer, ModelConfig
from src.lob.evals.quantized_model import benchmark_quantization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    transformer_config = ModelConfig.base()
    transformer_config.num_epochs = 30
    transformer_config.use_amp = True
    transformer_config.use_compile = False  # Set True if using GPU with torch 2.0+
    transformer_config.print_summary("Model Configuration")

    model = LOBTransformer(transformer_config).to(device)

    benchmark_quantization(model, transformer_config)