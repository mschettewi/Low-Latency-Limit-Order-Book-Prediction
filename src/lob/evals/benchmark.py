import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb

wandb_project = "HPML LOB"


@torch.no_grad()
def benchmark_inference(model, config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        num_runs=10000, batch_size=1, wandb_name=""):
    if device == torch.device("cuda"):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = model.to(device)
    model.eval()

    dummy = torch.randn(batch_size, config.seq_length, config.input_dim).to(device)

    # warm-up
    print("Warm up")
    for run in range(20):
        _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []

    for run in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append((time.perf_counter() - t0) * 1000)

    print(f"Mean latency: {np.mean(times):.2f} ms")
    print(f"Std latency: {np.std(times):.2f} ms")
    print(f"P50 latency: {np.percentile(times, 50):.2f} ms")
    print(f"P90 latency: {np.percentile(times, 90):.2f} ms")
    print(f"P95 latency: {np.percentile(times, 95):.2f} ms")

    plt.figure(figsize=(10, 6))
    plt.hist(times, bins=50)
    plt.title('Execution Time Distribution')
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')

    plt.savefig(f'logs/inference_time_distribution_{wandb_name}_batch_size_{batch_size}.png')

    return {
        "batch_size": batch_size,
        "mean_latency_ms": np.mean(times),
        "std_latency_ms": np.std(times),
        "p50_latency_ms": np.percentile(times, 50),
        "p90_latency_ms": np.percentile(times, 90),
        "p95_latency_ms": np.percentile(times, 95),
        "average_throughput_samples_per_sec": batch_size / (np.mean(times) / 1000.0)
    }


def benchmark_model(model, config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), wandb_group="",
                    wandb_name=""):
    "Returns latency and throughput"
    stats_df = pd.DataFrame()
    for batch_size in [1, 8, 256]:
        print(f"\n--- Benchmarking inference for batch size {batch_size} ---")
        stats = benchmark_inference(model=model, config=config, device=device, batch_size=batch_size, wandb_name=wandb_name)
        stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)

    wandb.init(project=wandb_project, name=wandb_name, group=wandb_group)
    table = wandb.Table(dataframe=stats_df)
    wandb.log({f"Inference_benchmark_{wandb_name}": table})
    wandb.finish()
