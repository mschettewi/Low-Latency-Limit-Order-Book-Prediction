import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from src.lob.models.transformer import LOBTransformer, ModelConfig
from src.lob.evals.profile import profile_inference


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_config = ModelConfig.base()
    transformer_config.num_epochs = 30
    transformer_config.use_amp = True
    transformer_config.use_compile = False  # Set True if using GPU with torch 2.0+
    transformer_config.print_summary("Model Configuration")

    model = LOBTransformer(transformer_config).to(device)

    # Create a dummy batch of size 1 (Latency Scenario)
    dummy_input = torch.randn(1, transformer_config.seq_length, transformer_config.input_dim).to(device)

    # Profile the Base Transformer
    profile_inference(model, dummy_input, device, trace_name="transformer_base")
