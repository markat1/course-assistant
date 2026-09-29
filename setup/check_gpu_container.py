from importlib.metadata import version

import torch

def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable inside the container.")
    
    device = torch.cuda.get_device_properties(0)

    print(f"SGLang: {version('sglang')}")
    print(f"Pytorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {device.name}")
    print(f"GPU memory: {device.total_memory / 1024**3:.1f} GiB")

    free, total = torch.cuda.mem_get_info()
    print(f"Visible memory limit: {total / 1024**3:.1f} GiB")

    matrix = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    result = matrix @ matrix

    if not torch.all(result == 32).item():
        raise SystemExit("BF16 matrix multiplication returned incorrect values.")

    print("PASS: CUDA allocation and BF16 matrix multiplication.")

if __name__ == "__main__":
    check_gpu()