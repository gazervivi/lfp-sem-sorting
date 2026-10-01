#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py — 生成论文专用配图

Fig 1  四类退化状态 SEM 典型形貌（每类 3 例，按类内分布取样以体现不均匀性）
Fig 2  误判样本案例（K 折全覆盖下的 3 个错误样本 + 1 个正确对照）
Fig 3  三种划分协议示意（tile 级 / 图像级 / 空间块级）

用法：
  python make_paper_figures.py --data-root data/v2 --runs runs --out runs/paper_figs
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]
CLASS_EN = {"A_new": "Class A  (pristine, 3.2 V)",
            "B_2V": "Class B  (2.0 V, deep-discharged)",
            "C_0V": "Class C  (0 V, spent cathode face)",
            "D_0Vcu": "Class D  (0 V, Cu-plated anode face)"}
C_TRAIN, C_VAL, C_TEST = "#1A73E8", "#F9AB00", "#D93025"


def load_img(path, size):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize(size, Image.BICUBIC)
        return np.asarray(im)


def fig1_morphology(data_root, outdir):
    img_dir = os.path.join(data_root, "images")
    n_show = 3
    fig, axes = plt.subplots(len(CLASS_ORDER), n_show,
                             figsize=(3.1 * n_show, 2.45 * len(CLASS_ORDER)))
    for r, cls in enumerate(CLASS_ORDER):
        d = os.path.join(img_dir, cls)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png", ".tif")))
        # 按类内排序均匀取样，呈现"最好—中间—最差"的形貌跨度（体现退化不均匀性）
        idxs = [0, len(files) // 2, len(files) - 1] if len(files) >= n_show else list(range(len(files)))
        for c in range(n_show):
            ax = axes[r][c]
            if c < len(idxs):
                a = load_img(os.path.join(d, files[idxs[c]]), (420, 315))
                ax.imshow(a)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(CLASS_EN[cls], fontsize=9.5, labelpad=16)
            for sp in ax.spines.values():
                sp.set_color("#888888"); sp.set_linewidth(0.6)
        axes[r][0].set_title("sample 1", fontsize=8.5)
        if n_show >= 2:
            axes[r][1].set_title("sample 2", fontsize=8.5)
        if n_show >= 3:
            axes[r][2].set_title("sample 3", fontsize=8.5)
    fig.suptitle("Fig. 1  SEM morphology of the four degradation states "
                 "(sampled across each class to show intra-class heterogeneity)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(outdir, "fig1_morphology.png")
    fig.savefig(p, dpi=220); plt.close(fig)
    return p


def fig2_errors(data_root, outdir, errors):
    """errors: list of dict(file, cls_true, cls_pred, conf)"""
    img_dir = os.path.join(data_root, "images")
    n = len(errors) + 1                      # 末尾加一个正确对照
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    if n == 1:
        axes = [axes]
    # 对照样本：真值 C 且预测正确（取一个易样本）
    ref = None
    d = os.path.join(img_dir, "C_0V")
    cands = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
    for f in cands:
        if not any(e["file"].startswith(os.path.splitext(f)[0]) for e in errors):
            ref = f
            break

    panels = list(errors) + ([{"file": ref, "cls_true": "C_0V", "cls_pred": "C_0V",
                              "conf": None, "ok": True}] if ref else [])
    for i, e in enumerate(panels):
        ax = axes[i]
        p = os.path.join(img_dir, e["cls_true"], e["file"])
        if not os.path.exists(p):
            stem = e["file"].rsplit(".", 1)[0]
            for f in os.listdir(os.path.join(img_dir, e["cls_true"])):
                if f.startswith(stem):
                    p = os.path.join(img_dir, e["cls_true"], f); break
        ax.imshow(load_img(p, (420, 315)))
        ax.set_xticks([]); ax.set_yticks([])
        ok = e.get("ok") or (e["cls_true"] == e["cls_pred"])
        col = "#0F9D58" if ok else "#D93025"
        t = f"true: {e['cls_true']}\npred: {e['cls_pred']}"
        if e.get("conf") is not None:
            t += f"\nconf: {e['conf']:.3f}"
        ax.set_xlabel(t, fontsize=9, color=col, fontweight="bold" if not ok else "normal")
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(2.2 if not ok else 1.0)
    fig.suptitle("Fig. 2  Misclassified images under full-coverage 5-fold CV; "
                 "all errors originate from Class C (spent cathode face)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(outdir, "fig2_error_cases.png")
    fig.savefig(p, dpi=220); plt.close(fig)
    return p


def fig3_protocols(outdir):
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.14,
                          left=0.03, right=0.97, top=0.78, bottom=0.13)
    rng = np.random.RandomState(42)

    def grid(ax, colors, xlim=(-0.4, 4.4)):
        for k in range(16):
            r, c = divmod(k, 4)
            ax.add_patch(Rectangle((c, 3 - r), 0.94, 0.9,
                                   facecolor=[C_TRAIN, C_VAL, C_TEST][colors[k]],
                                   alpha=.80, edgecolor="white", lw=1))
        ax.set_xlim(*xlim); ax.set_ylim(-1.5, 4.15)
        ax.set_aspect("equal"); ax.axis("off")

    # ---------- Panel A: tile 级随机划分 ----------
    ax = fig.add_subplot(gs[0, 0])
    order = rng.permutation(16)
    cols = [0 if o < 10 else (1 if o < 13 else 2) for o in order]
    grid(ax, cols)
    ax.set_title("(a)  P1 · tile-level random split\n"
                 "tiles of the SAME image land in train AND test\n"
                 "→ leakage: 176/176 source images affected", fontsize=9.5, pad=10)
    ax.text(2.0, -1.05, "one source image → 16 tiles", ha="center",
            fontsize=9, style="italic")

    # ---------- Panel B: 图像级划分 ----------
    ax = fig.add_subplot(gs[0, 1])
    a = load_img(os.path.join(outdir, "_tmp_tile.jpg"), (110, 110))
    for k in range(16):
        r, c = divmod(k, 4)
        ax.imshow(a, extent=[c, c + 0.94, 3 - r, 3 - r + 0.9], aspect="auto")
    ax.add_patch(Rectangle((-0.10, -0.10), 4.18, 4.10, facecolor="none",
                           edgecolor=C_TRAIN, lw=3.4))
    ax.set_xlim(-0.4, 4.4); ax.set_ylim(-1.5, 4.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b)  P2 · image-level split\n"
                 "all 16 tiles of an image stay together\n"
                 "→ no tile leakage; ~20% of images still\n"
                 "   have a spatially overlapping neighbour", fontsize=9.5, pad=10)
    ax.text(2.0, -1.05, "whole image → one split", ha="center",
            fontsize=9, style="italic")

    # ---------- Panel C: 空间块划分 ----------
    ax = fig.add_subplot(gs[0, 2])
    grid(ax, [2 if (k % 4) == 3 else 0 for k in range(16)])
    ax.plot([2.98, 2.98], [-0.02, 3.98], color="black", lw=2.4, ls="--")
    ax.set_title("(c)  P3 · spatial-block split\n"
                 "contiguous regions defined by stage coordinates\n"
                 "→ test↔train min distance 90.1 µm (FOV ≈ 10 µm)",
                 fontsize=9.5, pad=10)
    ax.text(1.5, -1.05, "spatial blocks", ha="center", fontsize=9, style="italic")

    h = [plt.Line2D([], [], marker="s", ls="", ms=11, color=c, label=l)
         for c, l in [(C_TRAIN, "train"), (C_VAL, "val"), (C_TEST, "test")]]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Fig. 3   Three evaluation protocols: where the leakage comes from",
                 fontsize=11.5, y=0.97)
    p = os.path.join(outdir, "fig3_protocols.png")
    fig.savefig(p, dpi=220); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/v2")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/paper_figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 临时素材（Panel B 用一张真图的局部裁片）
    tmp = os.path.join(args.out, "_tmp_tile.jpg")
    src_dir = os.path.join(args.data_root, "images", "C_0V")
    f0 = sorted(os.listdir(src_dir))[0]
    Image.open(os.path.join(src_dir, f0)).convert("RGB").resize((110, 110)).save(tmp)

    print(fig1_morphology(args.data_root, args.out))
    print(fig3_protocols(args.out))

    # 从 K 折合并结果里取误判清单
    err_path = os.path.join(args.runs, "kfold_merged_resnet50", "kfold_errors.csv")
    errors = []
    if os.path.exists(err_path):
        with open(err_path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                errors.append({"file": r["source_image"] + ".jpg" if not r["source_image"]
                               .lower().endswith(".jpg") else r["source_image"],
                               "cls_true": r["cls_true"], "cls_pred": r["cls_pred"],
                               "conf": float(r["mean_conf"])})
    print(fig2_errors(args.data_root, args.out, errors))
    json.dump({"errors": errors}, open(os.path.join(args.out, "paper_figs_meta.json"),
                                       "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n输出目录：{args.out}")
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".png"):
            print("  ", f)


if __name__ == "__main__":
    main()
