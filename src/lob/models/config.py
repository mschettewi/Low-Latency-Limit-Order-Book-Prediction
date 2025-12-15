from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Input dimensions
    seq_length: int = 100
    input_dim: int = 144
    num_classes: int = 3

    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_epochs: int = 5

    # Optimization flags
    use_amp: bool = True
    use_compile: bool = False  # Set to False for TPU compatibility

    @classmethod
    def small(cls):
        return cls(d_model=64, n_heads=4, n_layers=2, d_ff=256)

    @classmethod
    def base(cls):
        return cls()

    def print_summary(self, name: str):
        print(f"\n{name}:")
        print(f"  d_model={self.d_model}, heads={self.n_heads}, layers={self.n_layers}")
        print(f"  batch_size={self.batch_size}, lr={self.learning_rate}")
