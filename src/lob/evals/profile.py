# Profiling
import torch.profiler

def profile_inference(model, input_data, device, trace_name="inference_trace"):
    """
    Runs the PyTorch Profiler to capture GPU/CPU kernel usage.
    Exports a .json file you can view in chrome://tracing
    """
    print(f"Profiling {trace_name}...")
    model.eval()
    input_data = input_data.to(device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_data)

    # Profiling Context
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Run loop to trigger the schedule
        # (wait=1, warmup=1, active=3) -> Needs at least 5 steps
        for i in range(5):
            with torch.no_grad():
                _ = model(input_data)
            prof.step()

    print(f"Profiling complete. Logs saved to logs/profiler/")
    print(f"Download the json file from logs/profiler/ and open it in 'chrome://tracing' to visualize bottlenecks.")


