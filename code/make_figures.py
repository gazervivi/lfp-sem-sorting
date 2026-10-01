#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — E1 结果汇总：协议落差瀑布图 + 论文主表

产出（写到 --out）：
  table_master.csv / table_master.md   骨干 × 协议 主表
  fig_protocol_waterfall.png           协议落差瀑布图（本文最核心的一张图）
  fig_backbone_compare.png             骨干横向对比
  results_summary.json                 机器可读汇总

逻辑：
  tile 级精度  →  P1 与 P2 的落差 = 泄漏效应
  图像级精度（整图聚合）→  P2 的真实泛化性能
  一张图讲完"文献协议高估了多少"
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]


def image_level_accuracy(pred_csv):
    """从 tile 级预测 CSV 计算整图多数投票精度"""
    if not os.path.exists(pred_csv):
        return None, None, 0
    grp = defaultdict(list)
    with open(pred_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            grp[r["source_image"]].append(r)
    if not grp:
        return None, None, 0
    ok = 0
    consist = []
    for img, g in grp.items():
        gt = g[0]["cls_true"]
        votes = Counter(x["cls_pred"] for x in g)
        top, n = votes.most_common(1)[0]
        if len(votes) > 1 and n == votes.most_common(2)[1][1]:   # tie → 用概率均值
            avg = {c: sum(float(x[f"prob_{c[0]}"]) for x in g) / len(g) for c in CLASS_ORDER}
            top = max(avg, key=avg.get)
        ok += int(top == gt)
        consist.append(n / len(g))
    return ok / len(grp), sum(consist) / len(consist), len(grp)


def collect(runs_dir):
    rows = []
    for mf in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        d = json.load(open(mf, encoding="utf-8"))
        run_dir = os.path.dirname(mf)
        p = d["protocol"]
        if p.startswith("protocol_tilelevel"):
            proto = "P1_tilelevel"
        elif p.startswith("protocol_imagelevel"):
            proto = "P2_imagelevel"
        elif "kfold" in p:
            proto = "KF_crossval"
        elif "blocked" in p:
            proto = "P3_blocked"
        else:
            proto = p
        img_acc, img_cons, n_img = image_level_accuracy(
            os.path.join(run_dir, "predictions_test.csv"))
        rows.append({
            "run": os.path.basename(run_dir), "protocol": proto,
            "backbone": d["backbone"], "seed": d["seed"],
            "n_train": d.get("n_train"), "n_test": d.get("n_test"),
            "best_epoch": d.get("best_epoch"),
            "val_acc": d.get("best_val_acc"),
            "test_tile_acc": d.get("test_tile_acc"),
            "test_tile_f1": d.get("test_tile_f1_macro"),
            "test_image_acc": round(img_acc, 4) if img_acc is not None else None,
            "test_image_consistency": round(img_cons, 4) if img_cons is not None else None,
            "n_test_images": n_img,
            "minutes": d.get("total_min"),
        })
    return rows


def plot_waterfall(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    backbones = sorted({r["backbone"] for r in rows})
    fig, ax = plt.subplots(figsize=(math_w(backbones), 5.2))
    idx = np.arange(len(backbones))
    w = 0.26
    stage_names = ["P1 · tile-level split\n(literature practice, leaks)",
                   "P2 · image-level split\n(tile-level metric, leak-free)",
                   "P2 · image-level split\n(image-level vote, deployed)"]
    keys = ["test_tile_acc", "test_tile_acc", "test_image_acc"]
    protos = ["P1_tilelevel", "P2_imagelevel", "P2_imagelevel"]

    for j, (sn, k, pr) in enumerate(zip(stage_names, keys, protos)):
        vals = []
        for b in backbones:
            cand = [r for r in rows if r["backbone"] == b and r["protocol"] == pr and r[k] is not None]
            vals.append(max((c[k] for c in cand), default=np.nan))
        bars = ax.bar(idx + (j - 1) * w, [v * 100 for v in vals], w,
                      label=sn, color=["#9AA0A6", "#1A73E8", "#0F9D58"][j])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.6,
                        f"{v*100:.1f}", ha="center", fontsize=9)

    ax.set_xticks(idx); ax.set_xticklabels(backbones, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 104)
    ax.set_title("Protocol gap: tile-level random split vs. image-level split\n"
                 "(the drop quantifies how much the literature protocol over-estimates)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_protocol_waterfall.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def math_w(backbones):
    return max(6.5, 2.2 * len(backbones) + 2.5)


def plot_backbone(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    backbones = sorted({r["backbone"] for r in rows})
    fig, ax = plt.subplots(figsize=(math_w(backbones), 4.4))
    idx = np.arange(len(backbones))
    w = 0.34
    for j, (k, lab, col, pr) in enumerate([
            ("test_image_acc", "P2 image-level accuracy", "#0F9D58", "P2_imagelevel"),
            ("val_acc", "internal val accuracy", "#1A73E8", None)]):
        vals = []
        for b in backbones:
            cand = [r for r in rows if r["backbone"] == b and r[k] is not None
                    and (pr is None or r["protocol"] == pr)]
            vals.append(max((c[k] for c in cand), default=np.nan))
        bars = ax.bar(idx + (j - 0.5) * w, [v * 100 for v in vals], w, label=lab, color=col)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.6,
                        f"{v*100:.1f}", ha="center", fontsize=9)
    ax.set_xticks(idx); ax.set_xticklabels(backbones, fontsize=10)
    ax.set_ylim(50, 104); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Backbone comparison: internal validation vs. held-out generalization", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_backbone_compare.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def write_tables(rows, outdir):
    fields = ["run", "protocol", "backbone", "seed", "n_train", "n_test",
              "best_epoch", "val_acc", "test_tile_acc", "test_tile_f1",
              "test_image_acc", "test_image_consistency", "n_test_images", "minutes"]
    with open(os.path.join(outdir, "table_master.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    lines = ["| run | protocol | backbone | seed | best ep | val acc | test tile acc | "
             "test tile F1 | test image acc | min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v, n=4: f"{v:.{n}f}" if isinstance(v, float) else ("-" if v is None else str(v))
        lines.append(f"| {r['run']} | {r['protocol']} | {r['backbone']} | {r['seed']} | "
                     f"{f(r['best_epoch'],0)} | {f(r['val_acc'])} | {f(r['test_tile_acc'])} | "
                     f"{f(r['test_tile_f1'])} | {f(r['test_image_acc'])} | {f(r['minutes'],1)} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(outdir, "table_master.md"), "w", encoding="utf-8").write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="./runs")
    ap.add_argument("--out", default="./runs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = collect(args.runs_dir)
    if not rows:
        raise SystemExit(f"{args.runs_dir} 下没有 metrics.json，先跑 train_e1.py")
    print(f"收集到 {len(rows)} 个 run")

    md = write_tables(rows, args.out)
    print("\n" + md)
    p1 = plot_waterfall(rows, args.out)
    p2 = plot_backbone(rows, args.out)

    # 协议落差汇总（论文关键数字）
    gap = {}
    for b in sorted({r["backbone"] for r in rows}):
        a = [r for r in rows if r["backbone"] == b and r["protocol"] == "P1_tilelevel"
             and r["test_tile_acc"] is not None]
        c = [r for r in rows if r["backbone"] == b and r["protocol"] == "P2_imagelevel"
             and r["test_tile_acc"] is not None]
        if a and c:
            p1a = max(x["test_tile_acc"] for x in a)
            p2a = max(x["test_tile_acc"] for x in c)
            gap[b] = {"P1_tile_acc": round(p1a, 4), "P2_tile_acc": round(p2a, 4),
                      "leakage_gap_pp": round((p1a - p2a) * 100, 2)}
    json.dump({"runs": rows, "leakage_gap": gap},
              open(os.path.join(args.out, "results_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=== 泄漏落差（P1 tile 级 − P2 tile 级） ===")
    for b, g in gap.items():
        print(f"  {b:<18} {g['P1_tile_acc']*100:5.1f}% → {g['P2_tile_acc']*100:5.1f}%  "
              f"落差 {g['leakage_gap_pp']:+.2f} pp")
    print(f"\n图 → {p1}\n图 → {p2}\n表 → {args.out}/table_master.md")


if __name__ == "__main__":
    main()
