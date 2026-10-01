#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_kfold.py — 合并 K 折交叉验证的 test 预测，得到全样本覆盖的评测

为什么需要：单次划分的测试集只有 36 张图，Wilson 区间 ±8pp。
K 折交叉验证下每张图恰好被测试一次，合并后等价于 176 张图全部参与测试，
区间可收窄到 ±3pp 左右，per-class 指标也才有意义。

用法：
  python merge_kfold.py --runs /root/lfp/runs --pattern "kfold_f*" \
      --backbone resnet50 --out /root/lfp/runs/kfold_merged_resnet50
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/lfp/runs")
    ap.add_argument("--pattern", required=True, help='例如 "kfold_f*"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.runs, args.pattern, "predictions_test.csv")))
    if not files:
        raise SystemExit(f"未找到匹配的预测文件：{args.runs}/{args.pattern}")
    print(f"合并 {len(files)} 折：")
    rows = []
    for f in files:
        n = 0
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                r["_fold"] = os.path.basename(os.path.dirname(f))
                rows.append(r)
                n += 1
        print(f"  {os.path.basename(os.path.dirname(f)):<28} {n} tiles")

    # 去重（同一 tile 不应重复出现）
    seen, dedup = set(), []
    for r in rows:
        key = (r["_fold"], r["tile"])
        if key in seen:
            continue
        seen.add(key); dedup.append(r)
    rows = dedup
    print(f"合计 {len(rows)} 个 tile（去重后）")

    out_csv = os.path.join(args.out, "predictions_test.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # tile 级
    yt = [CLASS_ORDER.index(r["cls_true"]) for r in rows]
    yp = [CLASS_ORDER.index(r["cls_pred"]) for r in rows]
    k_ok = sum(a == b for a, b in zip(yt, yp))
    lo, hi = wilson(k_ok, len(yt))
    print(f"\n[tile 级] {k_ok}/{len(yt)} = {k_ok/len(yt)*100:.2f}%  "
          f"[95%CI {lo*100:.2f}–{hi*100:.2f}]")

    per = {}
    for c in CLASS_ORDER:
        idx = CLASS_ORDER.index(c)
        tp = sum(1 for a, b in zip(yt, yp) if a == idx and b == idx)
        fp = sum(1 for a, b in zip(yt, yp) if a != idx and b == idx)
        fn = sum(1 for a, b in zip(yt, yp) if a == idx and b != idx)
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        per[c] = {"precision": round(p, 4), "recall": round(rc, 4),
                  "f1": round(f1, 4), "support": sum(1 for a in yt if a == idx)}
        print(f"   {c:<7} P={p:.3f} R={rc:.3f} F1={f1:.3f} n={per[c]['support']}")

    # 图像级（多数投票）
    grp = defaultdict(list)
    for r in rows:
        grp[r["source_image"]].append(r)
    ok_img = 0
    errors = []
    for img, g in grp.items():
        gt = g[0]["cls_true"]
        votes = Counter(x["cls_pred"] for x in g)
        top, n = votes.most_common(1)[0]
        if len(votes) > 1 and n == votes.most_common(2)[1][1]:
            avg = {c: sum(float(x["prob_" + c[0]]) for x in g) / len(g) for c in CLASS_ORDER}
            top = max(avg, key=avg.get)
        if top == gt:
            ok_img += 1
        else:
            errors.append({"source_image": img, "cls_true": gt, "cls_pred": top,
                           "n_tiles": len(g), "vote_frac": round(n / len(g), 3),
                           "mean_conf": round(sum(float(x["confidence"]) for x in g) / len(g), 3)})
    lo2, hi2 = wilson(ok_img, len(grp))
    print(f"\n[图像级] {ok_img}/{len(grp)} = {ok_img/len(grp)*100:.2f}%  "
          f"[95%CI {lo2*100:.2f}–{hi2*100:.2f}]")
    print(f"错误样本 {len(errors)} 张：")
    for e in sorted(errors, key=lambda x: x["source_image"]):
        print(f"   {e['source_image']:<36} {e['cls_true']:<7} -> {e['cls_pred']:<7} "
              f"投票一致 {e['vote_frac']:.2f} 平均置信 {e['mean_conf']:.3f}")

    if errors:
        with open(os.path.join(args.out, "kfold_errors.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(errors[0].keys()))
            w.writeheader(); w.writerows(errors)

    json.dump({"n_folds": len(files), "n_tiles": len(yt),
               "tile_acc": round(k_ok / len(yt), 4),
               "tile_ci95": [round(lo, 4), round(hi, 4)],
               "n_images": len(grp), "image_acc": round(ok_img / len(grp), 4),
               "image_ci95": [round(lo2, 4), round(hi2, 4)],
               "per_class": per, "n_errors": len(errors)},
              open(os.path.join(args.out, "kfold_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n→ {out_csv}\n→ {args.out}/kfold_metrics.json")


if __name__ == "__main__":
    main()
