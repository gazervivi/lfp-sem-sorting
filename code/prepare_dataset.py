#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py — LFP 电池 SEM 数据集预处理与三级协议重划分

输入：2026.2.9-XH 原始图像文件夹（每张 jpg 配一份 SEM 元数据 txt）
输出：
  data_v2/images/{A_new,B_2V,C_0V,D_0Vcu}/  预处理后的整图
  data_v2/tiles/{A_new,B_2V,C_0V,D_0Vcu}/   切好的 tile
  data_v2/manifest_clean.csv                清洗后清单（含去重/下采样/倍率标记）
  data_v2/splits/protocol_tilelevel_seed{42,43,44}.csv     P1 tile 级随机划分（复现文献协议/泄漏）
  data_v2/splits/protocol_imagelevel_seed{42,43,44}.csv    P2 图像级划分（同电池未见视野）
  data_v2/splits/summary.json               划分统计

用法：
  python prepare_dataset.py \
      --src "/Users/nicolas/Downloads/原始图像/2026.2.9-XH" \
      --out "./data_v2" \
      --tile 256 --n-tiles 16 --seeds 42,43,44

设计要点：
  1) md5 去重：同内容多份只保留文件名最靠前的一张
  2) D 类 2560x1920 下采样至 1280x960，统一物理尺度（消除分辨率捷径特征）
  3) 类别映射 new->A, 2v->B, 0v->C, 0V-析铜->D
  4) P1 = 所有 tile 混池随机切分（文献通行做法，会泄漏）
     P2 = 按原图分组切分，tile 跟随其来源图（无泄漏）
     两个协议产出的 split csv 格式完全一致，训练脚本无需区分
  5) 所有随机过程固定种子，支持多种子重复实验
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict

from PIL import Image

CLASS_MAP = {
    "new": "A_new",        # A 类：全新电池正极
    "2v": "B_2V",          # B 类：2V 深度亏电（半新）
    "0v": "C_0V",          # C 类：0V 报废正极面
    "0V-析铜": "D_0Vcu",   # D 类：0V 报废铜箔析铜面
}
CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
TARGET_SIZE = (1280, 960)   # 统一下采样目标（宽, 高）


def parse_meta(txt_path):
    d = {}
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "," in line:
                    k, v = line.strip().split(",", 1)
                    d[k] = v
    return d


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan(src):
    """扫描原始文件夹，返回记录列表"""
    recs = []
    for root, _dirs, files in os.walk(src):
        cls_raw = os.path.basename(root)
        if cls_raw not in CLASS_MAP:
            continue
        for f in sorted(files):
            if not f.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(root, f)
            meta = parse_meta(os.path.splitext(p)[0] + ".txt")
            with Image.open(p) as im:
                w, h = im.size
            recs.append({
                "src": p, "file": f, "cls": CLASS_MAP[cls_raw], "cls_raw": cls_raw,
                "date": meta.get("DATE", ""), "time": meta.get("TIME", ""),
                "mag": meta.get("MAG", ""), "accvolt": meta.get("ACCVOLT", ""),
                "stgx": meta.get("STGX", ""), "stgy": meta.get("STGY", ""),
                "width": w, "height": h, "md5": md5_of(p),
                "is_dup": "", "downsampled": "", "note": "",
            })
    return recs


def dedup(recs):
    """同 md5 只保留文件名最靠前者，其余标记为重复"""
    by_md5 = defaultdict(list)
    for r in recs:
        by_md5[r["md5"]].append(r)
    for _m, group in by_md5.items():
        if len(group) > 1:
            group_sorted = sorted(group, key=lambda x: x["file"])
            for r in group_sorted[1:]:
                r["is_dup"] = "Y"
                r["note"] = "duplicate_of:" + group_sorted[0]["file"]
    return [r for r in recs if r["is_dup"] != "Y"]


def write_images(recs, out_dir):
    """下采样 + 另存预处理整图 + 拷贝元数据"""
    img_dir = os.path.join(out_dir, "images")
    for r in recs:
        dst_dir = os.path.join(img_dir, r["cls"])
        os.makedirs(dst_dir, exist_ok=True)
        with Image.open(r["src"]) as im:
            im = im.convert("RGB")
            if (im.width, im.height) != TARGET_SIZE:
                im = im.resize(TARGET_SIZE, Image.BICUBIC)
                r["downsampled"] = "Y"
            dst = os.path.join(dst_dir, r["file"])
            im.save(dst, quality=95)
        # 拷贝元数据 txt 一并留存
        src_txt = os.path.splitext(r["src"])[0] + ".txt"
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, os.path.splitext(dst)[0] + ".txt")
        r["proc_path"] = dst
    return recs


def cut_tiles(recs, out_dir, tile, n_tiles, seed=0):
    """每图切 n_tiles 个 tile（随机位置，固定种子）"""
    rng = random.Random(seed)
    tile_dir = os.path.join(out_dir, "tiles")
    n_skipped = 0
    for r in recs:
        dst_dir = os.path.join(tile_dir, r["cls"])
        os.makedirs(dst_dir, exist_ok=True)
        with Image.open(r["proc_path"]) as im:
            W, H = im.size
            if W < tile or H < tile:
                r["tiles"] = []
                n_skipped += 1
                continue
            names = []
            for t in range(n_tiles):
                x = rng.randint(0, W - tile)
                y = rng.randint(0, H - tile)
                patch = im.crop((x, y, x + tile, y + tile))
                stem = os.path.splitext(r["file"])[0]
                tname = f"{stem}_t{t:02d}.jpg"
                patch.save(os.path.join(dst_dir, tname), quality=95)
                names.append((tname, x, y))
            r["tiles"] = names
    if n_skipped:
        print(f"  [warn] {n_skipped} 张图小于 tile 尺寸，已跳过")
    return recs


def split_by_tile_level(recs, seed, ratios=(0.6, 0.2, 0.2)):
    """P1：所有 tile 混池随机划分（文献通行协议 —— 存在同一原图的 tile 跨集泄漏）"""
    pool = []
    for r in recs:
        for (tname, x, y) in r.get("tiles", []):
            pool.append({
                "tile": tname, "cls": r["cls"], "source_image": r["file"],
                "src_group": r["file"], "x": x, "y": y,
            })
    rng = random.Random(seed)
    rng.shuffle(pool)
    n = len(pool)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    for i, item in enumerate(pool):
        item["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return pool


def split_by_image_level(recs, seed, ratios=(0.6, 0.2, 0.2)):
    """P2：按原图分组划分，tile 跟随来源图（无泄漏）"""
    out = []
    rng = random.Random(seed)
    for cls in CLASS_ORDER:
        imgs = [r for r in recs if r["cls"] == cls]
        rng.shuffle(imgs)
        n = len(imgs)
        n_tr = max(1, int(round(n * ratios[0])))
        n_va = max(1, int(round(n * ratios[1])))
        if n_tr + n_va >= n:            # 保证每类至少 1 张进测试集
            n_tr, n_va = n - 2, 1

        for i, r in enumerate(imgs):
            sp = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            for (tname, x, y) in r.get("tiles", []):
                out.append({
                    "tile": tname, "cls": r["cls"], "source_image": r["file"],
                    "src_group": r["file"], "x": x, "y": y, "split": sp,
                })
    return out


def write_split(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["tile", "cls", "source_image", "src_group", "x", "y", "split"])
        w.writeheader()
        w.writerows(items)


def leak_stats(items):
    """统计有多少原图的 tile 跨集（泄漏规模）"""
    by_img = defaultdict(set)
    for it in items:
        by_img[it["source_image"]].add(it["split"])
    leaked = sum(1 for _i, s in by_img.items() if len(s) > 1)
    return leaked, len(by_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--n-tiles", type=int, default=16)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ratios", default="0.6,0.2,0.2",
                    help="train,val,test 比例（两套协议用同一比例，保证测试集规模一致可比）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ratios = tuple(float(x) for x in args.ratios.split(","))

    os.makedirs(args.out, exist_ok=True)
    print(f"[1/6] 扫描 {args.src}")
    recs = scan(args.src)
    print(f"      共 {len(recs)} 张；类别分布 {dict(Counter(r['cls'] for r in recs))}")

    print("[2/6] md5 去重")
    before = len(recs)
    recs = dedup(recs)
    print(f"      移除 {before - len(recs)} 张重复 → {len(recs)} 张")

    print("[3/6] 下采样统一物理尺度 + 另存")
    recs = write_images(recs, args.out)
    ds = sum(1 for r in recs if r["downsampled"] == "Y")
    print(f"      下采样 {ds} 张（原尺寸 > {TARGET_SIZE[0]}x{TARGET_SIZE[1]}）")

    print(f"[4/6] 切 tile（{args.tile}x{args.tile}, {args.n_tiles} 个/图）")
    recs = cut_tiles(recs, args.out, args.tile, args.n_tiles, seed=0)
    total_tiles = sum(len(r.get("tiles", [])) for r in recs)
    print(f"      共 {total_tiles} 个 tile")

    # 清洗后清单
    man_path = os.path.join(args.out, "manifest_clean.csv")
    fields = ["file", "cls", "cls_raw", "date", "time", "mag", "accvolt",
              "stgx", "stgy", "width", "height", "md5", "is_dup", "downsampled", "note"]
    with open(man_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print(f"      清单 → {man_path}")

    print("[5/6] 生成三级协议划分")
    summary = {"n_images": len(recs), "n_tiles": total_tiles, "protocols": {}}
    for seed in seeds:
        p1 = split_by_tile_level(recs, seed, ratios)
        p1_path = os.path.join(args.out, "splits", f"protocol_tilelevel_seed{seed}.csv")
        write_split(p1, p1_path)
        lk, tot = leak_stats(p1)
        c1 = Counter(it["split"] for it in p1)

        p2 = split_by_image_level(recs, seed, ratios)
        p2_path = os.path.join(args.out, "splits", f"protocol_imagelevel_seed{seed}.csv")
        write_split(p2, p2_path)
        lk2, tot2 = leak_stats(p2)
        c2 = Counter(it["split"] for it in p2)

        summary["protocols"][str(seed)] = {
            "tilelevel": {"path": p1_path, "counts": dict(c1),
                          "leaked_images": lk, "total_images": tot},
            "imagelevel": {"path": p2_path, "counts": dict(c2),
                           "leaked_images": lk2, "total_images": tot2},
        }
        print(f"  seed={seed}")
        print(f"    P1 tile 级: train/val/test = {c1.get('train',0)}/{c1.get('val',0)}/{c1.get('test',0)}"
              f" | 跨集泄漏原图 {lk}/{tot} ({lk/tot*100:.1f}%)")
        print(f"    P2 图像级: train/val/test = {c2.get('train',0)}/{c2.get('val',0)}/{c2.get('test',0)}"
              f" | 跨集泄漏原图 {lk2}/{tot2} ({lk2/tot2*100:.1f}%)")

    with open(os.path.join(args.out, "splits", "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("[6/6] 完成")
    print(f"\n输出目录: {args.out}")
    print("  images/     预处理整图（D 类已下采样统一尺度）")
    print("  tiles/      切好的 tile")
    print("  manifest_clean.csv   清洗后清单")
    print("  splits/     两种协议的划分文件 + summary.json")


if __name__ == "__main__":
    main()
