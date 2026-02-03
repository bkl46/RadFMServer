# accelerate_split_model.py
import torch
import torch.nn as nn
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
import time

def test_basic_split():
    """Test splitting a model across 2 GPUs using Accelerate"""
    print("=" * 60)
    print("Accelerate Model Splitting Test")
    print("=" * 60)
    
    # 1. Define a large model that won't fit on one GPU
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Create a model that's too large for one GPU
            self.embedding = nn.Embedding(10000, 2048)
            
            # Multiple large transformer-like blocks
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2048, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(4096, 2048),
                    nn.LayerNorm(2048)
                ) for _ in range(8)  # 8 large blocks
            ])
            
            self.output_head = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)  # Classification output
            )
        
        def forward(self, x):
            x = self.embedding(x)
            for block in self.blocks:
                x = block(x)
            return self.output_head(x.mean(dim=1))
    
    # 2. Initialize with empty weights (no memory allocated yet)
    print("Initializing model with empty weights...")
    with init_empty_weights():
        model = LargeModel()
    
    # 3. Check parameter sizes
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated memory (float32): {total_params * 4 / 1e9:.2f} GB")
    
    # 4. Create device map to split across GPUs
    print(f"\nAvailable GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for model splitting!")
        return
    
    # Get balanced memory allocation
    max_memory = get_balanced_memory(
        model,
        max_memory={0: "10GB", 1: "10GB"},  # Split equally
        no_split_module_classes=["Embedding", "LayerNorm"]
    )
    
    # 5. Create device map automatically
    print("\nCreating device map...")
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Embedding", "LayerNorm"],
        dtype="float32"
    )
    
    print("\nDevice Map:")
    for module_name, device in device_map.items():
        print(f"  {module_name}: {device}")
    
    # 6. Load model weights and dispatch to devices
    print("\nDispatching model to GPUs...")
    
    # First, we need to actually create the model with weights
    model = LargeModel()  # Real model with weights
    
    # Dispatch model across GPUs
    model = dispatch_model(
        model,
        device_map=device_map,
        main_device=0  # Main device for execution
    )
    
    return model, device_map

def test_inference(model, device_map):
    """Test inference with split model"""
    print("\n" + "=" * 60)
    print("Running Inference with Split Model")
    print("=" * 60)
    
    # Create sample input
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Move input to first GPU
    input_ids = input_ids.to("cuda:0")
    
    # Run inference multiple times
    num_runs = 5
    times = []
    
    with torch.no_grad():  # No gradient needed for inference
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Forward pass - Accelerate handles cross-GPU communication
            outputs = model(input_ids)
            
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i == 0:
                print(f"Output shape: {outputs.shape}")
    
    avg_time = sum(times) / len(times)
    print(f"\nInference Performance:")
    print(f"  Average time per inference: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {batch_size/avg_time:.2f} samples/sec")
    
    # Check memory usage
    print("\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved / {total:.2f} GB total")

if __name__ == "__main__":
    model, device_map = test_basic_split()
    if model:
        test_inference(model, device_map)
