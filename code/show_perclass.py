#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show_perclass.py — 打印指定 run 的分类别指标（P/R/F1/n）"""
import glob
import json
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/root/lfp/runs"
PATTERNS = sys.argv[2:] or ["blocked_*", "kfold_merged_*"]
CLS = ["A_new", "B_2V", "C_0V", "D_0Vcu"]

for pat in PATTERNS:
    for f in sorted(glob.glob(os.path.join(RUNS, pat, "metrics.json")) +
                    glob.glob(os.path.join(RUNS, pat, "kfold_metrics.json"))):
        d = json.load(open(f, encoding="utf-8"))
        tag = os.path.basename(os.path.dirname(f))
        if "kfold_metrics" in f:
            pc = d["per_class"]
            head = (f"tile={d['tile_acc']:.4f} CI{d['tile_ci95']}  "
                    f"image={d['image_acc']:.4f} CI{d['image_ci95']}")
        else:
            pc = d.get("test_per_class", {})
            head = f"val={d.get('best_val_acc'):.4f} test={d.get('test_tile_acc'):.4f}"
        print(f"{tag:<34} {head}")
        for c in CLS:
            v = pc.get(c)
            if not v:
                continue
            print("    {:<8} P={:.3f} R={:.3f} F1={:.3f} n={}".format(
                c, v["precision"], v["recall"], v["f1"], v["support"]))
        print()
