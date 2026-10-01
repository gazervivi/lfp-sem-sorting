#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_blocked_split.py — 空间块级划分（严格协议，第三层）

问题：图像级划分仍不干净。用台面坐标核查发现，同倍率下 FOV ≈ 10 µm，
      而 A 类最近邻距离最小 0.0 µm、中位 11.4 µm，B 类中位 16.6 µm
      —— 相邻"不同原图"实际拍的是同一块电极区域。图像级 test 图因此
      可能在空间上与训练图重叠，P2 性能仍被高估。

做法：沿主导空间轴排序后切成连续块，整块划入同一集合。
      因为排序后相邻即空间相邻，连续块 = 空间连通区域，
      这样 test 块与 train 块在样品台上是分离的。

分割方案（每类各自分块，保证折内类别均衡）：
      blocks 1-3 → train    block 4 → val（模型选择）    block 5 → test（评测）
      val 夹在中间，使 test 与 train 之间至少隔开一个块。

产出：data/v2/splits/blocked_imagelevel.csv（格式同 protocol_*.csv）

用法：
  python make_blocked_split.py --data-root data/v2 --k 5 --test-block 5
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]


def load_manifest(path):
    recs = defaultdict(list)
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                x, y = float(r["stgx"]), float(r["stgy"])
            except (ValueError, KeyError):
                continue
            stem = os.path.splitext(r["file"])[0]
            recs[r["cls"]].append({"stem": stem, "x": x, "y": y})
    return recs


def tiles_of(tiles_root, cls, stem):
    d = os.path.join(tiles_root, cls)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith(stem + "_t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/v2")
    ap.add_argument("--k", type=int, default=5, help="每类切成几个空间块")
    ap.add_argument("--test-block", type=int, default=5, help="第几块作为 test（1 起）")
    args = ap.parse_args()

    recs = load_manifest(os.path.join(args.data_root, "manifest_clean.csv"))
    tiles_root = os.path.join(args.data_root, "tiles")
    out_dir = os.path.join(args.data_root, "splits")
    os.makedirs(out_dir, exist_ok=True)

    rows, blocks_info = [], {}
    for c in CLASS_ORDER:
        g = recs.get(c, [])
        if not g:
            continue
        xs = [r["x"] for r in g]
        ys = [r["y"] for r in g]
        # 选择跨度更大的轴作为分块轴（空间偏移在该轴上更连续）
        axis = "x" if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else "y"
        g = sorted(g, key=lambda r: r[axis])
        n = len(g)
        # 等样本数切块
        edges = [round(i * n / args.k) for i in range(args.k + 1)]
        chunks = [g[edges[i]:edges[i + 1]] for i in range(args.k)]

        tb = args.test_block - 1
        vb = (tb - 1) % args.k          # val 紧邻 test 但位于 train 之外
        for bi, chunk in enumerate(chunks):
            if bi == tb:
                sp = "test"
            elif bi == vb:
                sp = "val"
            else:
                sp = "train"
            for r in chunk:
                for t in tiles_of(tiles_root, c, r["stem"]):
                    rows.append({"tile": t, "cls": c, "source_image": r["stem"],
                                 "src_group": r["stem"],
                                 "x": r["x"], "y": r["y"], "split": sp})
        blocks_info[c] = {"axis": axis, "n": n,
                          "chunk_sizes": [len(ch) for ch in chunks],
                          "test_block": args.test_block, "val_block": vb + 1}

    path = os.path.join(out_dir, "blocked_imagelevel.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["tile", "cls", "source_image",
                                           "src_group", "x", "y", "split"])
        w.writeheader(); w.writerows(rows)

    cnt = Counter(r["split"] for r in rows)
    byi = defaultdict(set)
    for r in rows:
        byi[r["source_image"]].add(r["split"])
    leak = sum(1 for _s, v in byi.items() if len(v) > 1)

    print(f"空间块级划分 → {path}")
    print(f"  tile 分布：{dict(cnt)}")
    print(f"  跨集泄漏原图：{leak}/{len(byi)}")
    for c, info in blocks_info.items():
        print(f"  [{c}] 分块轴={info['axis']}  每块图数={info['chunk_sizes']}  "
              f"test=块{info['test_block']} val=块{info['val_block']}")

    # 空间分离度自检：test 图到最近 train 图的距离
    import math
    tr_pts = [(r["x"], r["y"]) for r in rows if r["split"] == "train"]
    te_pts = sorted({(r["x"], r["y"]) for r in rows if r["split"] == "test"})
    dmin = []
    for t in te_pts:
        dmin.append(min(math.hypot(t[0] - s[0], t[1] - s[1]) for s in tr_pts) * 1000.0)
    if dmin:
        dmin.sort()
        print(f"\n  空间分离度：test 图到最近 train 图距离 "
              f"min={dmin[0]:.1f} median={dmin[len(dmin)//2]:.1f} max={dmin[-1]:.1f} µm")
        print(f"  （FOV≈10 µm，距离 >10 µm 即无空间重叠）")
        print(f"  无重叠的 test 图占比：{sum(1 for d in dmin if d > 10)/len(dmin)*100:.1f}%")
        blocks_info["_separation_um"] = {"min": round(dmin[0], 2),
                                         "median": round(dmin[len(dmin) // 2], 2)}
    json.dump(blocks_info, open(os.path.join(out_dir, "blocked_summary.json"),
                                "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
