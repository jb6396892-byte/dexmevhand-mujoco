"""Verify that the existing dexmv PyTorch installation can train on CUDA."""

import sys

import torch


def main():
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        print("CUDA unavailable: check the NVIDIA driver with nvidia-smi.", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(device))
    weights = torch.nn.Linear(64, 32).to(device)
    optimizer = torch.optim.Adam(weights.parameters(), lr=1e-3)
    inputs = torch.randn(16, 64, device=device)
    loss = weights(inputs).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    print("CUDA forward/backward/optimizer: OK; loss = {:.6f}".format(loss.item()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
