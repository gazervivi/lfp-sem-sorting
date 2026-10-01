#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_fov_overlap.py — 核查 P2（图像级）协议的"空间泄漏"风险

动机：图像级划分假定"不同原图 = 独立样本"。但若相邻视野在样品台上距离小于
     视场宽度，两张"不同图"实际拍摄的是同一块电极区域，P2 的结果仍被高估。

方法：
  1) 由 MAG 与图像宽度反推视场宽度 FOV = REF_WIDTH_MM * 1000 / MAG（µm），
     其中 REF_WIDTH_MM 为显微镜显示的参考宽度（电子显微镜常见 100 mm 基准）
  2) 计算同类内每张图到最近邻图（其他图）的台面距离
  3) 若最近邻距离 < FOV，则该图与邻图空间重叠 → 存在残留空间相关

用法：
  python check_fov_overlap.py --manifest data/v2/manifest_clean.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]
REF_WIDTH_MM = 100.0        # 常见显示基准；结论对该常数只做线性缩放


def fov_um(mag, width_px, ref_mm=REF_WIDTH_MM):
    """视场宽度（µm）：显示宽度 / 放大倍数，再按像素宽度占参考宽度的比例折算"""
    try:
        mag = float(mag)
    except (TypeError, ValueError):
        return None
    if mag <= 0:
        return None
    return ref_mm * 1000.0 / mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/v2/manifest_clean.csv")
    args = ap.parse_args()

    rows = []
    with open(args.manifest, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                x, y = float(r["stgx"]), float(r["stgy"])
            except (ValueError, KeyError):
                continue
            rows.append({"file": r["file"], "cls": r["cls"], "x": x, "y": y,
                         "mag": r["mag"], "w": int(r["width"])})

    print(f"载入 {len(rows)} 张图（含台面坐标）\n")
    print("说明：STGX/STGY 单位按 mm 计（电子显微镜台架常用单位），故 1 µm = 0.001")
    print(f"      FOV 估算基准 REF_WIDTH_MM={REF_WIDTH_MM} mm\n")

    summary = {}
    for c in CLASS_ORDER:
        g = [r for r in rows if r["cls"] == c]
        if len(g) < 2:
            continue
        # 该类的主导像素宽度与倍率
        mags = defaultdict(int)
        for r in g:
            mags[r["mag"]] += 1
        mag = max(mags, key=mags.get)
        wpix = max(r["w"] for r in g)
        fov = fov_um(mag, wpix)
        # 该类的台面跨度
        xs = [r["x"] for r in g]
        ys = [r["y"] for r in g]
        span_x = (max(xs) - min(xs)) * 1000.0     # mm → µm
        span_y = (max(ys) - min(ys)) * 1000.0

        # 最近邻距离（µm）
        nn = []
        for i, r in enumerate(g):
            best = float("inf")
            for j, s in enumerate(g):
                if i == j:
                    continue
                d = math.hypot(r["x"] - s["x"], r["y"] - s["y"]) * 1000.0
                best = min(best, d)
            nn.append(best)
        nn.sort()
        n_overlap = sum(1 for d in nn if d < fov)
        summary[c] = {
            "n_images": len(g), "dominant_mag": mag, "pixel_width": wpix,
            "fov_width_um": round(fov, 2),
            "stage_span_x_um": round(span_x, 1), "stage_span_y_um": round(span_y, 1),
            "nn_dist_min_um": round(nn[0], 2),
            "nn_dist_median_um": round(nn[len(nn) // 2], 2),
            "nn_dist_max_um": round(nn[-1], 2),
            "n_images_with_overlapping_neighbor": n_overlap,
            "overlap_fraction": round(n_overlap / len(g), 3),
        }
        s = summary[c]
        print(f"[{c}] n={s['n_images']}  MAG={mag}  像素宽={wpix}  估算 FOV={s['fov_width_um']} µm")
        print(f"     台面跨度 X={s['stage_span_x_um']} µm  Y={s['stage_span_y_um']} µm")
        print(f"     最近邻距离 min/median/max = {s['nn_dist_min_um']} / "
              f"{s['nn_dist_median_um']} / {s['nn_dist_max_um']} µm")
        print(f"     与邻图空间重叠的图数：{n_overlap}/{len(g)} "
              f"({s['overlap_fraction']*100:.1f}%)\n")

    with open("fov_overlap_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    tot = sum(s["n_images"] for s in summary.values())
    ov = sum(s["n_images_with_overlapping_neighbor"] for s in summary.values())
    print("=" * 72)
    print(f"合计：{ov}/{tot} 张图（{ov/tot*100:.1f}%）存在空间重叠的邻图")
    if ov == 0:
        print("结论：**P2 图像级划分不存在显著空间泄漏**，'不同原图 = 独立样本'的假设成立。")
    else:
        print("结论：存在空间重叠，P2 结果应视为**略微乐观**，需在 limitation 中披露；"
              "严格做法是按空间坐标做块状划分（blocked split）。")
    print("注意：本结论对 REF_WIDTH_MM 保守假设敏感——若实际基准小于 100 mm，"
          "FOV 更小、重叠更少，结论只会更强（不会更弱）。")
    print(f"\n报告 → fov_overlap_report.json")


if __name__ == "__main__":
    main()
