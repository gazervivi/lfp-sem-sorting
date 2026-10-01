#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_kfold.py — 生成图像级 K 折交叉验证划分（提升小样本统计强度）

动机：单一 80/20 划分下测试集只有 36 张图，Wilson 区间宽达 ±8 个百分点，
     per-class 指标基本没有统计意义。K 折交叉验证让每张图都被测试一次，
     等价于用全部 176 张图作为测试集，区间显著收窄。

产出：data/v2/splits/kfold{k}_f{i}_imagelevel.csv（k=折数，i=1..k）
      格式与 protocol_*.csv 完全一致，train_e1.py 可直接消费。

用法：
  python make_kfold.py --data-root data/v2 --k 5 --val-frac 0.12 --seed 42
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]


def list_images(tiles_root):
    """从 tiles 目录反推原图（文件名形如 <stem>_tNN.jpg）"""
    imgs = defaultdict(set)
    for cls in CLASS_ORDER:
        d = os.path.join(tiles_root, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue
            stem = f.rsplit("_t", 1)[0]
            imgs[cls].add(stem)
    return {c: sorted(v) for c, v in imgs.items()}


def tiles_of(tiles_root, cls, stem):
    d = os.path.join(tiles_root, cls)
    return sorted([f for f in os.listdir(d) if f.startswith(stem + "_t")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.12,
                    help="每折从训练图里再切出的验证图比例（用于选最优 epoch）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tiles_root = os.path.join(args.data_root, "tiles")
    out_dir = os.path.join(args.data_root, "splits")
    os.makedirs(out_dir, exist_ok=True)

    imgs = list_images(tiles_root)
    rng = random.Random(args.seed)
    folds = {c: [[] for _ in range(args.k)] for c in CLASS_ORDER}
    shuffled = {}
    for c in CLASS_ORDER:
        lst = imgs[c][:]
        rng.shuffle(lst)          # 打乱后才用于分折与选验证集
        shuffled[c] = lst
        for i, stem in enumerate(lst):
            folds[c][i % args.k].append(stem)      # 轮转发牌，各类别折内数量均衡

    summary = {}
    for fi in range(args.k):
        rows = []
        for c in CLASS_ORDER:
            test = set(folds[c][fi])
            # 关键：验证集必须从【已打乱】的剩余样本里取，
            # 否则 rest[:n_val] 会按文件名（= 拍摄时间）顺序取到连续视野，引入选择偏差
            rest = [s for s in shuffled[c] if s not in test]
            n_val = max(1, int(round(len(rest) * args.val_frac)))
            vset = set(rest[:n_val])
            for stem in imgs[c]:
                sp = "test" if stem in test else ("val" if stem in vset else "train")
                for t in tiles_of(tiles_root, c, stem):
                    rows.append({"tile": t, "cls": c, "source_image": stem,
                                 "src_group": stem,
                                 "x": -1, "y": -1, "split": sp})
        path = os.path.join(out_dir, f"kfold{args.k}_f{fi+1}_imagelevel.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["tile", "cls", "source_image",
                                               "src_group", "x", "y", "split"])
            w.writeheader(); w.writerows(rows)
        cnt = Counter(r["split"] for r in rows)
        # 泄漏自检
        leak = 0
        byi = defaultdict(set)
        for r in rows:
            byi[r["source_image"]].add(r["split"])
        leak = sum(1 for _s, v in byi.items() if len(v) > 1)
        summary[f"fold{fi+1}"] = {"path": path, "tiles": dict(cnt),
                                  "leaked_images": leak, "n_images": len(byi)}
        print(f"  fold {fi+1}: tile {dict(cnt)} | 跨集泄漏原图 {leak}/{len(byi)}")

    import json
    json.dump(summary, open(os.path.join(out_dir, f"kfold{args.k}_summary.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n共生成 {args.k} 折，目录：{out_dir}")
    print("说明：每张原图恰好作为 test 出现一次 → 合并 k 折 test 结果 = 全部 "
          f"{sum(len(v) for v in imgs.values())} 张图的完整评测")


if __name__ == "__main__":
    main()
