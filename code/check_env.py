#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_env.py — 服务器环境体检（实验前第一步）

检查：Python / PyTorch / CUDA / GPU 型号与显存 / 关键依赖 / 磁盘空间 / 数据完整性。
不修改任何环境，只做只读探测。
"""

import importlib
import os
import shutil
import subprocess
import sys

OK = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"

REQUIRED = {
    "torch": None,
    "torchvision": None,
    "PIL": "pillow",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "tqdm": "tqdm",
}
OPTIONAL = {"timm": "timm", "seaborn": "seaborn", "captum": "captum"}


def line(c, msg):
    print(f"{c} {msg}")


def main():
    print("=" * 68)
    print(" LFP SEM 实验环境体检")
    print("=" * 68)

    # ---- 1. Python ----
    print("\n[1] Python 运行时")
    v = sys.version_info
    line(OK if v >= (3, 9) else FAIL, f"Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")

    # ---- 2. 依赖 ----
    print("\n[2] 关键依赖")
    missing = []
    for mod, pip_name in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            line(OK, f"{mod:<14} {ver}")
        except Exception:
            line(FAIL, f"{mod:<14} 缺失 → pip install {pip_name}")
            missing.append(pip_name)
    for mod, pip_name in OPTIONAL.items():
        try:
            m = importlib.import_module(mod)
            line(OK, f"{mod:<14} {getattr(m, '__version__', '?')}  (可选)")
        except Exception:
            line(WARN, f"{mod:<14} 未安装（可选）→ pip install {pip_name}")

    # ---- 3. GPU ----
    print("\n[3] GPU / CUDA")
    try:
        import torch
        line(OK, f"torch {torch.__version__}")
        line(OK, f"CUDA 编译版本 {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            line(OK, f"可用 GPU 数 {n}")
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                total = p.total_memory / 1024 ** 3
                line(OK, f"  GPU{i}: {p.name} | {total:.1f} GB | SM {p.major}.{p.minor} "
                         f"| {p.multi_processor_count} SM")
                if total < 9:
                    line(WARN, f"  GPU{i} 显存 {total:.1f} GB，Swin 需 AMP + batch<=16 + 梯度累积")
            # 一次极小的实算校验
            try:
                x = torch.randn(8, 3, 224, 224, device="cuda")
                import torchvision
                m = torchvision.models.resnet18(weights=None).cuda().eval()
                with torch.no_grad():
                    y = m(x)
                line(OK, f"smoke test 通过，输出形状 {tuple(y.shape)}")
            except Exception as e:
                line(FAIL, f"smoke test 失败：{e}")
        else:
            line(FAIL, "CUDA 不可用！检查驱动 / 是否装成 CPU 版 torch")
    except Exception as e:
        line(FAIL, f"torch 不可用：{e}")
        missing.append("torch torchvision")

    # ---- 4. 磁盘 ----
    print("\n[4] 磁盘空间")
    for path in [".", os.path.expanduser("~"), "/tmp"]:
        try:
            t, u, f = shutil.disk_usage(path)
            gb = lambda x: x / 1024 ** 3
            c = OK if gb(f) > 20 else WARN
            line(c, f"{path:<24} 总 {gb(t):6.1f}G  已用 {gb(u):6.1f}G  可用 {gb(f):6.1f}G")
        except Exception as e:
            line(WARN, f"{path}: {e}")
    line(WARN, "预估需求：数据+中间产物 ~5G，checkpoint+图 ~3G，建议可用空间 > 20G")

    # ---- 5. CPU / 内存 ----
    print("\n[5] CPU / 内存")
    try:
        line(OK, f"CPU 逻辑核心 {os.cpu_count()}")
        if hasattr(os, "sysconf"):
            rss = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
            line(OK, f"内存 {rss:.1f} GB")
        line(OK, "DataLoader workers 建议 = min(8, CPU核数//2)")
    except Exception as e:
        line(WARN, str(e))

    # ---- 6. 汇总 ----
    print("\n" + "=" * 68)
    if missing:
        print(" 待安装（复制执行）：")
        print("   pip install " + " ".join(sorted(set(missing))))
    else:
        print(" 环境就绪，可以直接进入 prepare_dataset.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
