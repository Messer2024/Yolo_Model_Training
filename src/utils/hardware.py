"""
硬件环境检测与显存探测工具 (Hardware Detection)
"""
from typing import Dict, Any, List
import platform
import os


def detect_hardware() -> Dict[str, Any]:
    """
    探测当前系统的硬件环境 (CPU, CUDA GPU, MPS, 显存大小等)
    :return: 包含硬件信息的字典
    """
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Unknown CPU",
        "is_cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "default_device": "cpu",
        "primary_gpu_name": "None",
        "primary_gpu_vram_gb": 0.0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["is_cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            count = torch.cuda.device_count()
            info["device_count"] = count
            info["default_device"] = "0"

            for i in range(count):
                name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                info["devices"].append({
                    "id": str(i),
                    "name": name,
                    "vram_gb": round(total_mem, 2)
                })

            if count > 0:
                info["primary_gpu_name"] = info["devices"][0]["name"]
                info["primary_gpu_vram_gb"] = info["devices"][0]["vram_gb"]

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["default_device"] = "mps"
            info["devices"].append({
                "id": "mps",
                "name": "Apple Silicon (MPS)",
                "vram_gb": 0.0
            })
    except Exception as e:
        info["error"] = str(e)

    return info
