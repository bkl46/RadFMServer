# test_simple_load.py
import torch
import time

print("Testing simple checkpoint load...")
start = time.time()

try:
    # Just try to load the checkpoint to CPU RAM
    print("Attempting to load checkpoint to CPU...")
    checkpoint = torch.load('./Language_files/pytorch_model.bin', map_location='cpu')
    print(f"✓ Success! Loaded checkpoint in {time.time()-start:.1f}s")
    
    # Check size
    total_size = 0
    for key, tensor in checkpoint.items():
        total_size += tensor.numel() * tensor.element_size()
    
    print(f"Checkpoint memory usage: {total_size / 1e9:.2f}GB")
    print(f"Number of parameters: {sum(t.numel() for t in checkpoint.values()) / 1e9:.2f}B")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
