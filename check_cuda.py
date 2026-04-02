# By dreamraster · dreaMSCend
#!/usr/bin/env python3
"""Check PyTorch CUDA installation and configuration."""

import torch
import sys

def check_pytorch_cuda():
    """Verify PyTorch and CUDA setup."""
    print("=" * 50)
    print("PyTorch CUDA Check")
    print("=" * 50)

    # Print Python version
    print(f"Python: {sys.version}")

    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is AVAILABLE!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available")
        print("PyTorch was built without CUDA support or drivers are not installed.")

    print("=" * 50)

    # Test with a simple tensor
    try:
        x = torch.rand(10, 10)
        y = torch.rand(10, 10)
        z = torch.matmul(x, y)
        print("Basic tensor operations: OK")
        print(f"Tensor shape: {z.shape}")
    except Exception as e:
        print(f"Error with tensor operations: {e}")

    # Test CUDA operations if available
    if torch.cuda.is_available():
        try:
            x_cuda = torch.rand(10, 10).cuda()
            y = torch.mm(x_cuda, x_cuda)
            print("CUDA matrix multiplication: OK")
            print(f"CUDA tensor device: {y.device}")
        except Exception as e:
            print(f"Error with CUDA operations: {e}")

    # List available devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")

if __name__ == "__main__":
    check_pytorch_cuda()
