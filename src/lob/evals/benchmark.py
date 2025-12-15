import time

import numpy as np
import torch


def benchmark_inference(model, config, device, num_runs=100, batch_size=1):
    """Benchmark inference latency and throughput"""
    model = model.to(device)
    model.eval()

    dummy = torch.randn(batch_size, config.seq_length, config.input_dim).to(device)

    # Warm-up
    print(f"Warming up GPU (batch_size={batch_size})...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_runs} inference iterations...")
    times = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)

    print(f"\n{'=' * 70}")
    print(f"Inference Benchmark (batch_size={batch_size})")
    print(f"{'=' * 70}")
    print(f"Mean latency:   {times.mean():.2f} ms")
    print(f"Std latency:    {times.std():.2f} ms")
    print(f"P50 latency:    {np.percentile(times, 50):.2f} ms")
    print(f"P90 latency:    {np.percentile(times, 90):.2f} ms")
    print(f"P95 latency:    {np.percentile(times, 95):.2f} ms")
    print(f"Min latency:    {times.min():.2f} ms")
    print(f"Max latency:    {times.max():.2f} ms")
    print(f"Throughput:     {1000 * batch_size / times.mean():.1f} samples/sec")

    return times
