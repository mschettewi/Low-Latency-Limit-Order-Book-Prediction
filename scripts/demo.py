import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lob.demo.stream import visualize
from src.lob.data.dataset import get_datasets
from src.lob.models.transformer import LOBTransformer, ModelConfig
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig.base()
    model = LOBTransformer(config)
    ckpt_path = os.path.join("checkpoints", "transformer_balanced.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    model.eval()

    train_dataset, val_dataset, test_dataset = get_datasets()

    obj = train_dataset.lob

    visualize(model=model, obj=obj)
