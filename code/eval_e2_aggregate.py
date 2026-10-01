#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_e2_aggregate.py — E2：tile 聚合决策 + 选择性分类（拒判）分析

输入：train_e1.py 产出的 predictions_{val,test}.csv（tile 级预测 + 四类概率）
      也兼容 predict/ 目录下的旧格式 CSV（中文表头，tile 名内嵌原图名）
输出（写到 --out 目录）：
  e2_aggregation_compare.csv   tile 级 / 多数投票 / 概率均值 三种决策的整图精度
  e2_risk_coverage.csv         拒判曲线的原始数据（阈值-覆盖率-精度）
  e2_reject_summary.json       最优工作点（约束覆盖率下的最高精度）
  e2_metrics.json              分类别指标 + 混淆矩阵
  fig_e2_aggregation.png       三种决策精度对比 + risk-coverage 曲线
  e2_per_image.csv             每张原图的聚合预测明细（供人工复核误判样本）

核心逻辑：
  1) 以"原图"为决策单元：同一张 SEM 图的所有 tile 共同决定该图的最终判级
  2) 三种聚合规则：单 tile 直判（基线）/ 多数投票 / 概率均值（soft voting）
  3) 拒判：整图置信度（聚合后最大概率）低于阈值 → 判为"不确定"，
     业务含义 = 送拆解冶炼线（保守分流），不进修复线
  4) 同时给出不等代价视角：漏判报废(把 C/D 判成 A/B)与误杀良品(A/B 判成 C/D)代价不同，
     输出两者的计数，供论文讨论章节使用
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]
LETTER2FULL = {"A": "A_new", "B": "B_2V", "C": "C_0V", "D": "D_0Vcu"}
FULL2LETTER = {v: k for k, v in LETTER2FULL.items()}
# 业务语义：A=全新、B=2V 中度退化（可进修复线），C=0V 报废正极、D=0V 析铜（须送冶炼）
REPAIRABLE = {"A_new", "B_2V"}      # 可进修复线
SCRAP = {"C_0V", "D_0Vcu"}          # 须送冶炼线

# 旧格式中文表头映射
LEGACY_MAP = {"图片": "tile", "真实": "cls_true", "预测": "cls_pred",
              "置信度": "confidence", "正确": "correct",
              "A": "prob_A", "B": "prob_B", "C": "prob_C", "D": "prob_D"}


def _norm_cls(v):
    """统一为完整类名：'A' / 'A_new' / '不确定_A' 都归一为 'A_new'"""
    v = (v or "").strip()
    if v.startswith("不确定"):
        v = v.split("_")[-1]
    if v in LETTER2FULL:
        return LETTER2FULL[v]
    return v


def load_predictions(path):
    """读取 tile 级预测，归一化列名，返回记录列表"""
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"空文件：{path}")

    cols = list(rows[0].keys())
    is_legacy = "图片" in cols
    out = []
    for r in rows:
        if is_legacy:
            tile_raw = r["图片"]
            # 旧格式： "B\\2V_20260109,161900.jpg_t03" → 原图名在 .jpg_/.tif_ 之前
            stem = tile_raw
            for sep in (".jpg_", ".tif_", ".jpeg_", ".png_"):
                if sep in stem:
                    stem = stem.split(sep)[0]
                    break
            tile = tile_raw
            source = os.path.basename(stem)
            probs = [float(r["A"]), float(r["B"]), float(r["C"]), float(r["D"])]
            gt = _norm_cls(r["真实"]); pr = _norm_cls(r["预测"])
            conf = float(r["置信度"])
        else:
            tile = r["tile"]
            source = r.get("source_image") or tile.rsplit("_t", 1)[0]
            probs = [float(r["prob_A"]), float(r["prob_B"]),
                     float(r["prob_C"]), float(r["prob_D"])]
            gt = _norm_cls(r["cls_true"]); pr = _norm_cls(r["cls_pred"])
            conf = float(r["confidence"])
        if gt not in CLASS_ORDER:
            continue
        out.append({"tile": tile, "source_image": source, "cls_true": gt,
                    "cls_pred": pr, "confidence": conf, "probs": probs})
    return out


def agg_by_image(recs):
    """按原图聚合，返回每图记录（含三种决策结果）"""
    groups = defaultdict(list)
    for r in recs:
        groups[r["source_image"]].append(r)
    out = []
    for img, g in groups.items():
        gt = g[0]["cls_true"]
        # 决策 1：单 tile 直判（取整图内第一个 tile，作为无聚合基线）
        d1 = g[0]["cls_pred"]
        # 决策 2：多数投票（tie 时取该图内平均概率最高者）
        votes = Counter(x["cls_pred"] for x in g)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            avg_all = [sum(x["probs"][i] for x in g) / len(g) for i in range(4)]
            d2 = CLASS_ORDER[avg_all.index(max(avg_all))]
        else:
            d2 = top[0][0]
        # 决策 3：概率均值（soft voting）
        avg = [sum(x["probs"][i] for x in g) / len(g) for i in range(4)]
        p3 = avg.index(max(avg))
        d3 = CLASS_ORDER[p3]
        # 图内一致性
        consist = top[0][1] / len(g)
        # 拒判置信度：投票一致率 × 均值概率（综合证据强度与一致性）
        reject_conf = consist * avg[p3]
        out.append({
            "source_image": img, "cls_true": gt, "n_tiles": len(g),
            "pred_single": d1, "correct_single": int(d1 == gt),
            "pred_vote": d2, "correct_vote": int(d2 == gt),
            "pred_soft": d3, "correct_soft": int(d3 == gt),
            "vote_consistency": round(consist, 4),
            "prob_mean_max": round(avg[p3], 4),
            "reject_conf": round(reject_conf, 4),
            "cls_pred_soft": d3,
        })
    return sorted(out, key=lambda x: x["source_image"])


def risk_coverage(per_img, thresholds):
    """拒判曲线：阈值越高，只保留高置信图 → 覆盖率下降、精度上升"""
    rows = []
    n = len(per_img)
    for th in thresholds:
        cov = [r for r in per_img if r["reject_conf"] >= th]
        if not cov:
            rows.append({"threshold": th, "coverage": 0.0, "n_covered": 0,
                         "selective_acc": None, "n_rejected": n,
                         "rejected_with_scrap_true": 0, "rejected_with_repairable_true": 0})
            continue
        acc = sum(r["correct_soft"] for r in cov) / len(cov)
        rej = [r for r in per_img if r["reject_conf"] < th]
        rows.append({
            "threshold": round(th, 3), "coverage": round(len(cov) / n, 4),
            "n_covered": len(cov), "selective_acc": round(acc, 4),
            "n_rejected": len(rej),
            "rejected_with_scrap_true": sum(1 for r in rej if r["cls_true"] in SCRAP),
            "rejected_with_repairable_true": sum(1 for r in rej if r["cls_true"] in REPAIRABLE),
        })
    return rows


def per_class_metrics(per_img, key):
    stat = {}
    for c in CLASS_ORDER:
        tp = sum(1 for r in per_img if r["cls_true"] == c and r[key] == c)
        fp = sum(1 for r in per_img if r["cls_true"] != c and r[key] == c)
        fn = sum(1 for r in per_img if r["cls_true"] == c and r[key] != c)
        n = sum(1 for r in per_img if r["cls_true"] == c)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        stat[c] = {"precision": round(p, 4), "recall": round(rc, 4),
                   "f1": round(f1, 4), "support": n}
    acc = sum(r[key] == r["cls_true"] for r in per_img) / len(per_img)
    f1m = sum(stat[c]["f1"] for c in CLASS_ORDER) / 4
    cm = [[sum(1 for r in per_img if r["cls_true"] == t and r[key] == p)
           for p in CLASS_ORDER] for t in CLASS_ORDER]
    return {"accuracy": round(acc, 4), "macro_f1": round(f1m, 4),
            "per_class": stat, "confusion_matrix": cm}


def plot_fig(per_img, rc, outdir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    # (a) 三种决策的整图精度
    names = ["single tile", "majority vote", "prob. mean"]
    vals = [sum(r["correct_single"] for r in per_img) / len(per_img),
            sum(r["correct_vote"] for r in per_img) / len(per_img),
            sum(r["correct_soft"] for r in per_img) / len(per_img)]
    bars = ax[0].bar(names, [v * 100 for v in vals],
                     color=["#9AA0A6", "#1A73E8", "#0F9D58"])
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v * 100 + 0.4,
                   f"{v*100:.1f}%", ha="center", fontsize=10)
    ax[0].set_ylim(0, 105); ax[0].set_ylabel("image-level accuracy (%)")
    ax[0].set_title(f"(a) Aggregation rules  (n={len(per_img)} images)")
    ax[0].grid(alpha=.3, axis="y")

    # (b) risk-coverage
    xs = [r["coverage"] * 100 for r in rc if r["selective_acc"] is not None]
    ys = [r["selective_acc"] * 100 for r in rc if r["selective_acc"] is not None]
    ax[1].plot(xs, ys, "o-", color="#D93025")
    ax[1].set_xlabel("coverage (%)"); ax[1].set_ylabel("selective accuracy (%)")
    ax[1].set_title("(b) Risk-coverage curve"); ax[1].grid(alpha=.3)
    ax[1].invert_xaxis()

    # (c) 置信度分布（按真实类别着色）
    colors = {"A_new": "#1A73E8", "B_2V": "#0F9D58",
              "C_0V": "#F9AB00", "D_0Vcu": "#D93025"}
    bins = [i / 20 for i in range(21)]
    for c in CLASS_ORDER:
        vs = [r["reject_conf"] for r in per_img if r["cls_true"] == c]
        if vs:
            ax[2].hist(vs, bins=bins, alpha=.55, label=c, color=colors[c])
    ax[2].set_xlabel("aggregated confidence"); ax[2].set_ylabel("# images")
    ax[2].set_title("(c) Confidence distribution by true class")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.suptitle(tag, fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_e2_aggregation.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions_test.csv（或旧格式 CSV）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--thresh-step", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recs = load_predictions(args.pred)
    per_img = agg_by_image(recs)
    tag = args.tag or os.path.basename(os.path.dirname(args.pred))

    print(f"tile 级样本 {len(recs)} → 整图 {len(per_img)} 张")
    print(f"类别分布: {dict(Counter(r['cls_true'] for r in per_img))}")

    # 1) 三种决策对比
    cmp_rows = []
    for key, name in (("pred_single", "single_tile"), ("pred_vote", "majority_vote"),
                      ("pred_soft", "probability_mean")):
        m = per_class_metrics(per_img, key)
        cmp_rows.append({"rule": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         **{f"{c}_{k}": m["per_class"][c][k]
                            for c in CLASS_ORDER for k in ("precision", "recall", "f1")}})
        print(f"  {name:<18} acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    with open(os.path.join(args.out, "e2_aggregation_compare.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader(); w.writerows(cmp_rows)

    # 2) 拒判曲线
    ths = [round(i * args.thresh_step, 3) for i in range(int(1 / args.thresh_step) + 1)]
    rc = risk_coverage(per_img, ths)
    with open(os.path.join(args.out, "e2_risk_coverage.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rc[0].keys()))
        w.writeheader(); w.writerows(rc)
    # 最优工作点：优先覆盖率 >= 80% 时精度最高
    cand = [r for r in rc if r["selective_acc"] is not None and r["coverage"] >= 0.8]
    best80 = max(cand, key=lambda r: r["selective_acc"]) if cand else None
    bestany = max((r for r in rc if r["selective_acc"] is not None),
                  key=lambda r: (r["selective_acc"], r["coverage"]))
    print(f"  覆盖率>=80% 最优工作点: th={best80['threshold']} "
          f"cov={best80['coverage']:.3f} acc={best80['selective_acc']:.4f}" if best80
          else "  覆盖率>=80% 无可行工作点")
    print(f"  精度最高点: th={bestany['threshold']} cov={bestany['coverage']:.3f} "
          f"acc={bestany['selective_acc']:.4f}")

    # 3) 不等代价统计（业务口径）
    d3 = per_class_metrics(per_img, "pred_soft")
    missed_scrap = sum(1 for r in per_img if r["cls_true"] in SCRAP and r["cls_pred_soft"] in REPAIRABLE)
    killed_good = sum(1 for r in per_img if r["cls_true"] in REPAIRABLE and r["cls_pred_soft"] in SCRAP)
    # 拒判后的风险（全部拒判样本按"送冶炼"处理 → 报废漏判降为 0，代价是好料被误送）
    rej_rows = [r for r in per_img if r["reject_conf"] < (best80["threshold"] if best80 else 0.0)]
    print(f"  不等代价：漏判报废 {missed_scrap} 张 | 误杀良品 {killed_good} 张")

    summary = {"input": args.pred, "tag": tag,
               "n_tiles": len(recs), "n_images": len(per_img),
               "aggregation": cmp_rows, "soft_metrics": d3,
               "risk_coverage": {"best_at_coverage_ge_80": best80, "best_overall": bestany},
               "asymmetric_cost": {"missed_scrap_into_repair_line": missed_scrap,
                                   "repairable_killed_to_scrap_line": killed_good,
                                   "n_rejected_at_best80": len(rej_rows)},
               "images_by_true": dict(Counter(r["cls_true"] for r in per_img))}
    with open(os.path.join(args.out, "e2_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # 4) 每图明细（人工复核误判样本用）
    fields = list(per_img[0].keys())
    with open(os.path.join(args.out, "e2_per_image.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(per_img)

    # 5) 出图
    p = plot_fig(per_img, rc, args.out, tag)
    print(f"  图 → {p}")

    errs = [r for r in per_img if not r["correct_soft"]]
    print(f"\n错误样本 {len(errs)} 张（供物理机理分析）：")
    for r in errs:
        print(f"  {r['source_image']:<38} 真实={r['cls_true']:<7} "
              f"预测={r['cls_pred_soft']:<7} 聚合置信={r['reject_conf']:.3f} "
              f"图内一致性={r['vote_consistency']:.2f} ({r['n_tiles']} tiles)")
    print(f"\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
