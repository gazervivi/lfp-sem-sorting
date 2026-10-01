#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_e4_tsne.py — E4：特征空间可视化 + 跨模型错误一致性（误差物理闭环的证据链）

产出：
  e4_features_<backbone>.npz       每张原图的聚合特征 + 标签
  e4_tsne_<backbone>.csv           t-SNE 二维坐标
  e4_cross_model_errors.csv        跨模型错误交集（本征歧义的量化证据）
  fig_e4_tsne.png                  左：按真实类别着色；右：按 tile 预测一致性着色
  fig_e4_error_overlap.png         跨模型错误 Venn/条形图

用法：
  python eval_e4_tsne.py \
     --data-root ./data_v2 \
     --split ./data_v2/splits/protocol_imagelevel_seed42.csv \
     --ckpt ./runs/e1_P2_resnet50_s42/best.pt \
     --runs-dir ./runs \
     --out ./runs/e4_analysis
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]
CLS_SHORT = {"A_new": "A (fresh)", "B_2V": "B (2V)", "C_0V": "C (0V)", "D_0Vcu": "D (0V-Cu)"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_split(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def find_head(model):
    """定位分类头（取其输入 = 倒数第二层特征）"""
    for name in ("fc", "head", "classifier"):
        if hasattr(model, name):
            m = getattr(model, name)
            return m[-1] if isinstance(m, nn.Sequential) else m
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


def build_backbone(backbone, num_classes=4, weights=None):
    """与 train_e1.py 保持一致的构建逻辑（仅结构，不含权重）"""
    import torchvision.models as tvm
    if backbone == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "swin_t":
        m = tvm.swin_t(weights=None); m.head = nn.Linear(m.head.in_features, num_classes)
    elif backbone == "swin_s":
        m = tvm.swin_s(weights=None); m.head = nn.Linear(m.head.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        import timm
        name = {"dinov2": "vit_small_patch14_dinov2.lvd142m"}.get(backbone, backbone)
        m = timm.create_model(name, pretrained=False, num_classes=num_classes)
    return m


@torch.no_grad()
def extract_features(model, items, tiles_root, head, img_size=224, bs=64, device="cuda"):
    """返回 (per-image 特征, per-image 标签, per-image tile 预测数组)"""
    feats, preds, labels = [], [], []
    buf_f, buf_p, buf_l, buf_img = [], [], [], []
    model.eval()
    for it in items:
        p = os.path.join(tiles_root, it["cls"], it["tile"])
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
            a = np.asarray(im).astype(np.float32) / 255.0
        m, s = a.mean(), a.std() + 1e-6          # 与训练一致：逐图标准化
        a = (a - m) / s
        a = (a - 0.0) / 1.0
        buf_f.append(torch.from_numpy(a.transpose(2, 0, 1).copy()))
        buf_l.append(CLASS_ORDER.index(it["cls"]))
        buf_img.append(it["source_image"])
        if len(buf_f) >= bs:
            _, _, _ = _flush(model, buf_f, buf_l, buf_img, head, feats, preds, labels, device)
            buf_f, buf_p, buf_l, buf_img = [], [], [], []
    if buf_f:
        _flush(model, buf_f, buf_l, buf_img, head, feats, preds, labels, device)

    # 按原图聚合：特征取均值，预测取众数
    grp = defaultdict(list)
    for f, pr, lb, im in zip(feats, preds, labels, [x for x in _IMG_CACHE]):
        grp[im].append((f, pr, lb))
    out = []
    for im, g in grp.items():
        F = np.mean([x[0] for x in g], axis=0)
        votes = Counter(x[1] for x in g)
        out.append({"source_image": im, "label": g[0][2], "feat": F,
                    "vote_top": votes.most_common(1)[0][0],
                    "vote_consistency": votes.most_common(1)[0][1] / len(g),
                    "preds": [x[1] for x in g]})
    return out


_IMG_CACHE = []


def _flush(model, buf_f, buf_l, buf_img, head, feats, preds, labels, device):
    x = torch.stack(buf_f).to(device)
    cap = {}
    h = head.register_forward_hook(lambda m, inp, out: cap.__setitem__("in", inp[0].detach()))
    logits = model(x)
    h.remove()
    logits = logits.float().cpu().numpy()
    F = cap["in"].float().cpu().numpy().reshape(len(buf_f), -1)
    for i in range(len(buf_f)):
        feats.append(F[i]); preds.append(CLASS_ORDER[int(logits[i].argmax())])
        labels.append(buf_l[i]); _IMG_CACHE.append(buf_img[i])
    return None, None, None


def cross_model_errors(runs_dir, outdir):
    """跨模型错误交集：不同骨干在同一样本上是否犯同样的错"""
    per_run = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", "predictions_test.csv"))):
        run = os.path.basename(os.path.dirname(f))
        err = set()
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["correct"] == "F":
                    err.add(r["source_image"])
        per_run[run] = err
    if not per_run:
        print("[WARN] 未找到 predictions_test.csv，跳过跨模型分析")
        return None
    # 按 (backbone, protocol) 归并：同一张图在所有骨干上都错 = 高度怀疑本征歧义
    img_fail = defaultdict(set)
    for run, err in per_run.items():
        for im in err:
            img_fail[im].add(run)
    rows = []
    for im, runs in sorted(img_fail.items(), key=lambda x: -len(x[1])):
        rows.append({"source_image": im, "n_runs_failed": len(runs),
                     "runs": ";".join(sorted(runs))})
    with open(os.path.join(outdir, "e4_cross_model_errors.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 图：共同错误数分布
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cnt = Counter(r["n_runs_failed"] for r in rows)
    n_runs = len(per_run)
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(cnt)
    ax.bar([str(k) for k in ks], [cnt[k] for k in ks], color="#D93025")
    for i, k in enumerate(ks):
        ax.text(i, cnt[k] + 0.1, str(cnt[k]), ha="center")
    ax.set_xlabel(f"# of runs failing on the same image  (total runs = {n_runs})")
    ax.set_ylabel("# source images")
    ax.set_title("Cross-model error overlap — shared failures indicate intrinsic ambiguity")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_e4_error_overlap.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"跨模型共同错误样本 {len(rows)} 个，其中被 {n_runs} 个 run 同时判错的有 "
          f"{cnt.get(n_runs, 0)} 个 → {p}")
    return per_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs-dir", default="./runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--perplexity", type=float, default=15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone")
    print(f"载入 {args.ckpt}  backbone={backbone} epoch={ck.get('epoch')}")
    model = build_backbone(backbone, len(CLASS_ORDER))
    model.load_state_dict(ck["model"]); model = model.to(device)

    items = read_split(args.split)
    items = [it for it in items if it["split"] in ("train", "val", "test")]
    tiles_root = os.path.join(args.data_root, "tiles")
    head = find_head(model)
    print(f"特征提取：{len(items)} 个 tile（分类头输入维度捕获中）")

    global _IMG_CACHE
    _IMG_CACHE = []
    per_img = extract_features(model, items, tiles_root, head,
                               args.img_size, 64, device)
    X = np.stack([r["feat"] for r in per_img])
    y = np.array([r["label"] for r in per_img])
    cons = np.array([r["vote_consistency"] for r in per_img])
    names = [r["source_image"] for r in per_img]
    print(f"整图样本 {len(per_img)} 张，特征维度 {X.shape[1]}")

    np.savez_compressed(os.path.join(args.out, f"e4_features_{backbone}.npz"),
                        X=X, y=y, names=np.array(names), consistency=cons)

    from sklearn.manifold import TSNE
    per = min(args.perplexity, max(5, (len(X) - 1) / 3))
    Z = TSNE(n_components=2, perplexity=per, init="pca",
             learning_rate="auto", random_state=42).fit_transform(X)
    with open(os.path.join(args.out, f"e4_tsne_{backbone}.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["source_image", "tsne_x", "tsne_y", "class", "consistency"])
        for i, n in enumerate(names):
            w.writerow([n, Z[i, 0], Z[i, 1], CLASS_ORDER[y[i]], cons[i]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A_new": "#1A73E8", "B_2V": "#0F9D58", "C_0V": "#F9AB00", "D_0Vcu": "#D93025"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for i in range(2):
        for c in CLASS_ORDER:
            m = y == CLASS_ORDER.index(c)
            if m.sum() == 0: continue
            if i == 0:
                ax[i].scatter(Z[m, 0], Z[m, 1], label=f"{c} (n={m.sum()})",
                              s=42, alpha=.85, color=colors[c], edgecolors="white", linewidths=.5)
            else:
                sc = ax[i].scatter(Z[m, 0], Z[m, 1], c=cons[m], cmap="viridis",
                                   vmin=0.3, vmax=1.0, s=42, alpha=.9,
                                   edgecolors="white", linewidths=.5)
        ax[i].set_title("(a) colored by true class" if i == 0
                        else "(b) colored by intra-image tile consistency")
        ax[i].set_xticks([]); ax[i].set_yticks([]); ax[i].grid(alpha=.25)
    ax[0].legend(fontsize=8, loc="best")
    fig.colorbar(sc, ax=ax[1], shrink=.85, label="tile vote consistency")
    fig.suptitle(f"Feature-space embedding of image-level SEM representations ({backbone})", fontsize=12)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_e4_tsne.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"t-SNE 图 → {p}")

    # 类间/类内距离（重叠程度的量化）
    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(X, y)
        print(f"轮廓系数（类别可分性，越高越可分）：{sil:.4f}")
    except Exception as e:
        sil = None
    json.dump({"backbone": backbone, "n_images": len(per_img), "n_features": int(X.shape[1]),
               "silhouette": sil,
               "consistency_by_class": {c: float(np.mean(cons[y == CLASS_ORDER.index(c)]))
                                        for c in CLASS_ORDER}},
              open(os.path.join(args.out, f"e4_summary_{backbone}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    cross_model_errors(args.runs_dir, args.out)
    print(f"\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
