"""
First proof point of the project: confirm the container can see the host GPU
through the NVIDIA Container Toolkit before anything else gets built on top.

Run with:
    docker run --rm --gpus all triton-drift-ops/nvidia-smoke-test
"""

import shutil
import subprocess
import sys


def check_nvidia_smi() -> bool:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found in container — GPU passthrough is not configured")
        return False

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        print("nvidia-smi failed:")
        print(result.stderr)
        return False

    print(result.stdout)
    return True


if __name__ == "__main__":
    ok = check_nvidia_smi()
    sys.exit(0 if ok else 1)
