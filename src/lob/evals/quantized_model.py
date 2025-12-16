import os
import time

import torch.quantization
from torch.utils.data import DataLoader

from src.lob.models.transformer import LOBTransformer


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f'Size (MB): {size_mb:.2f}')
    os.remove('temp.p')


def benchmark_quantization(model, config):
    print("\n" + "=" * 70)
    print("DYNAMIC QUANTIZATION BENCHMARK (CPU SCENARIO)")
    print("=" * 70)

    # 1. Prepare Baseline Model on CPU
    # We must move to CPU because Pytorch Dynamic Quantization is a CPU-specific optimization
    model_cpu = LOBTransformer(config).to("cpu")
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    print("Original Model (FP32) on CPU:")
    print_model_size(model_cpu)

    # Benchmark Baseline CPU Latency
    dummy_input_cpu = torch.randn(1, config.seq_length, config.input_dim).to("cpu")
    print("Benchmarking FP32 CPU Latency...")

    # Warmup
    for _ in range(5):
        _ = model_cpu(dummy_input_cpu)

    # Measure
    t0 = time.time()
    for _ in range(50):
        _ = model_cpu(dummy_input_cpu)
    avg_latency_fp32 = (time.time() - t0) / 50 * 1000
    print(f"FP32 CPU Latency: {avg_latency_fp32:.2f} ms")

    # 2. Apply Dynamic Quantization
    print("\nApplying Selective Quantization...")

    model_cpu.input_projection = torch.quantization.quantize_dynamic(
        model_cpu.input_projection,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    model_cpu.fc = torch.quantization.quantize_dynamic(
        model_cpu.fc,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Rename for clarity
    quantized_model = model_cpu

    print("Quantized Model (Int8 - Hybrid):")
    print_model_size(quantized_model)

    # 3. Benchmark Quantized Latency
    print("Benchmarking Int8 CPU Latency...")
    # Warmup
    for _ in range(5):
        _ = quantized_model(dummy_input_cpu)

    # Measure
    t0 = time.time()
    for _ in range(50):
        _ = quantized_model(dummy_input_cpu)
    avg_latency_int8 = (time.time() - t0) / 50 * 1000

    print(f"Int8 CPU Latency: {avg_latency_int8:.2f} ms")
    print(f"\nSpeedup: {avg_latency_fp32 / avg_latency_int8:.2f}x")

    return quantized_model


# Helper function to calculate accuracy
def get_model_accuracy(model, dataset, device="cpu", max_samples=1000):
    # We limit samples to 1000 to keep it fast on CPU
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

            if total >= max_samples:
                break

    return correct / total
