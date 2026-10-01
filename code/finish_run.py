#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finish_run.py — 用已保存的 best.pt 补做推理与指标（不需要重新训练）

使用场景：训练已完成并保存了 best.pt / curves.png / per_epoch_metrics.csv，
但最后的推理环节中断（例如 torch.load 在 PyTorch 2.6+ 下的 weights_only 变更）。
本脚本读取同目录的 config.json 还原实验配置，重放推理部分。

直接复用 train_e1.py 里的实现，保证与正常流程产出的文件完全一致。

用法：
  python finish_run.py --run /root/lfp/runs/e1_P1_resnet50_s42 [--force]
"""

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_e1 as T          # noqa: E402


def best_from_log(run_dir):
    p = os.path.join(run_dir, "per_epoch_metrics.csv")
    if not os.path.exists(p):
        return None, None, None
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    if not rows:
        return None, None, None
    b = max(rows, key=lambda r: float(r["val_acc"]))
    return float(b["val_acc"]), float(b["val_f1_macro"]), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--force", action="store_true", help="即使 metrics.json 已存在也重做")
    args = ap.parse_args()
    run = args.run.rstrip("/")

    cfg_path = os.path.join(run, "config.json")
    ck_path = os.path.join(run, "best.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(ck_path)):
        raise SystemExit(f"缺少 config.json 或 best.pt：{run}")
    if os.path.exists(os.path.join(run, "metrics.json")) and not args.force:
        print(f"[skip] {os.path.basename(run)} 已有 metrics.json")
        return

    cfg = json.load(open(cfg_path, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = cfg["backbone"]
    print(f"[finish] {os.path.basename(run)}  backbone={backbone} device={device}")

    items = T.read_split(cfg["split"])
    tiles_root = os.path.join(cfg["data_root"], "tiles")
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    te = [i for i in items if i["split"] == "test"]

    gnorm = not cfg.get("no_grayscale_norm", False)
    isz = cfg.get("img_size", 224)
    bs = cfg.get("bs", 32)
    wk = cfg.get("workers", 4)

    ds_va = T.TileDataset(va, tiles_root, isz, train=False, grayscale_norm=gnorm)
    ds_te = T.TileDataset(te, tiles_root, isz, train=False, grayscale_norm=gnorm)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=wk)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=wk)

    model, _note = T.build_model(backbone, 4, weights=cfg.get("weights"))
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device)

    bva, bf1, n_ep = best_from_log(run)
    summary = {
        "backbone": backbone, "seed": cfg.get("seed"),
        "protocol": os.path.basename(cfg["split"]),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "best_epoch": ck.get("epoch"),
        "best_val_acc": bva, "best_val_f1": bf1, "epochs_run": n_ep,
        "config": cfg,
    }

    for name, ds, dl in (("val", ds_va, dl_va), ("test", ds_te, dl_te)):
        if len(ds) == 0:
            continue
        rows = T.predict(model, dl, device)
        out_rows = []
        for i, y, prob in rows:
            it = ds.items[i]
            p = int(prob.argmax())
            out_rows.append({
                "tile": it["tile"], "source_image": it["source_image"],
                "cls_true": T.CLASS_ORDER[y], "cls_pred": T.CLASS_ORDER[p],
                "confidence": round(float(prob[p]), 6),
                "correct": "T" if p == y else "F",
                "prob_A": round(float(prob[0]), 6), "prob_B": round(float(prob[1]), 6),
                "prob_C": round(float(prob[2]), 6), "prob_D": round(float(prob[3]), 6),
            })
        with open(os.path.join(run, f"predictions_{name}.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        ys = [r[1] for r in rows]
        ps = [int(r[2].argmax()) for r in rows]
        acc, f1m, pc = T.compute_metrics(ys, ps)
        cm = T.confusion_matrix(ys, ps, labels=list(range(4))) if hasattr(T, "confusion_matrix") \
            else __import__("sklearn.metrics", fromlist=["confusion_matrix"]) \
            .confusion_matrix(ys, ps, labels=list(range(4)))
        T.plot_cm(cm, os.path.join(run, f"cm_{name}.png"),
                  f"{backbone} seed{cfg.get('seed')} {name} tile-level")
        summary[f"{name}_tile_acc"] = round(float(acc), 4)
        summary[f"{name}_tile_f1_macro"] = round(float(f1m), 4)
        summary[f"{name}_per_class"] = pc
        print(f"  [{name}] acc={acc:.4f}  macro-F1={f1m:.4f}  n_tiles={len(rows)}")

    summary["total_min"] = None
    json.dump(summary, open(os.path.join(run, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  → {run}/metrics.json 已生成")


if __name__ == "__main__":
    main()
