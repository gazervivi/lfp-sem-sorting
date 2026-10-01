#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_e3_cam.py — E3：可解释性 + 因果量化（HiResCAM / Grad-CAM++ + AOPC）

为什么用 HiResCAM 而不是普通 Grad-CAM：
  Grad-CAM 对激活做全局平均池化，会把同一特征图上的不同空间位置混在一起，
  定位精度差且可能"高亮"模型实际没用的区域。HiResCAM 用 grad ⊙ activation 的
  逐元素乘积，对 CNN 局部可解释性更忠实（本文用它验证"模型看的是颗粒团聚/
  裸露衬底/析铜沉积"，定位错位会直接毁掉这条论证）。

AOPC（Average Over Perturbation Curve）：
  按显著性排序逐步抹除像素（替换为图像均值），测量目标类概率的下降。
  AOPC 高 = 显著性区域是模型决策的真实依赖；AOPC ≈ 0 = 显著性图在胡说。
  这是把"可解释性图好看"升级为"因果可信"的关键量化。

产出（写到 --out）：
  cam_overlays/<image>_<tile>_<backbone>.png   原图 | 显著性 | 叠加
  e3_cam_index.csv                              样本、真实/预测类、CAM 峰值位置
  e3_aopc.csv / fig_e3_aopc.png                 逐样本 AOPC 与分布
  e3_summary.json                               按类别聚合的 AOPC

用法：
  python eval_e3_cam.py --data-root data/v2 \
      --split data/v2/splits/protocol_imagelevel_seed42.csv \
      --ckpt runs/e1_P2_resnet50_s42/best.pt \
      --out runs/e3_resnet50 --per-class 2 --split-name test
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]


def build_backbone(backbone, num_classes=4):
    import torchvision.models as tvm
    if backbone == "resnet50":
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "swin_t":
        m = tvm.swin_t(weights=None); m.head = nn.Linear(m.head.in_features, num_classes)
    elif backbone == "swin_s":
        m = tvm.swin_s(weights=None); m.head = nn.Linear(m.head.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        import timm
        m = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    return m


def target_layer(model, backbone):
    """选取最后一个具备空间分辨率的卷积/特征层"""
    if backbone.startswith("resnet"):
        return model.layer4[-1]
    if backbone.startswith("efficientnet"):
        return model.features[-1]
    if backbone.startswith("swin"):
        return model.features[-1][-1].norm1 if hasattr(model.features[-1][-1], "norm1") \
               else model.features[-1][-1]
    # timm 通用回退
    for name in ("norm", "norm_pre", "blocks"):
        if hasattr(model, name):
            m = getattr(model, name)
            return m if not isinstance(m, nn.Sequential) else m[-1]
    raise SystemExit("无法自动定位目标层，请手动指定")


def hirescam(model, layer, x, class_idx):
    """HiResCAM：grad ⊙ activation 逐元素相乘后对通道求和"""
    store = {}

    def fwd(_m, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, class_idx].backward()
    h.remove()

    act, grad = store["act"][0], store["grad"][0]           # (C,H,W)
    cam = torch.relu((grad * act).sum(dim=0))                # HiResCAM
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy(), logits.detach()


def _margin(z, c):
    """目标类对数几率与最强竞争类之差（不受 softmax 饱和影响）"""
    others = torch.cat([z[:c], z[c + 1:]])
    return float(z[c] - others.max())


def _deletion_curve(model, x, class_idx, order, steps, frac_max, device, m0, p0):
    """按给定像素顺序逐级抹除，返回 (margin 下降曲线, prob 下降曲线) 的均值"""
    H, W = None, None
    total = order.size
    mc, pc = [0.0], [0.0]
    for k in range(1, steps + 1):
        frac = k / steps * frac_max
        n = int(total * frac)
        idx = order[:n]
        xp = x.clone()
        flat = xp.view(1, 3, -1)
        flat[0, :, torch.as_tensor(idx, device=xp.device, dtype=torch.long)] = xp.mean()
        with torch.no_grad():
            zk = model(xp)[0]
        pc.append(p0 - torch.softmax(zk, 0)[class_idx].item())
        mc.append(m0 - _margin(zk, class_idx))
    return float(np.mean(mc)), float(np.mean(pc))


def aopc(model, x, class_idx, cam, steps=10, device="cuda", frac_max=0.7, seed=42):
    """AOPC：按显著性顺序 vs 随机顺序逐级抹除像素，测量目标类 logit margin 下降。

    关键设计——必须有随机对照：
      仅看"抹除显著区导致多少下降"无法解释，因为 SEM 纹理类图像对任意像素缺失
      都可能稳健。正确判据是 **显著区删除的下降 − 随机删除的下降**：
        > 0 → 显著性图确实抓住了模型真正依赖的区域（忠实）
        ≈ 0 → 模型决策分散在全图统计量上，不存在局部"诊断区域"
        < 0 → 显著性图与模型决策无关（不可信）
    """
    with torch.no_grad():
        z0 = model(x)[0]
    p0 = float(torch.softmax(z0, 0)[class_idx])
    m0 = _margin(z0, class_idx)

    cam_order = np.argsort(-cam.ravel())
    rng = np.random.RandomState(seed)
    rand_order = rng.permutation(cam.size)

    m_cam, p_cam = _deletion_curve(model, x, class_idx, cam_order,
                                   steps, frac_max, device, m0, p0)
    m_rand, p_rand = _deletion_curve(model, x, class_idx, rand_order,
                                     steps, frac_max, device, m0, p0)
    return {"aopc_margin": m_cam, "aopc_prob": p_cam,
            "aopc_margin_random": m_rand, "aopc_prob_random": p_rand,
            "aopc_gain": m_cam - m_rand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--per-class", type=int, default=2, help="每类抽样图数（含误判优先）")
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--aopc-steps", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "cam_overlays"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone")
    model = build_backbone(backbone).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    layer = target_layer(model, backbone)
    print(f"{args.ckpt}  backbone={backbone}  目标层={type(layer).__name__}")

    with open(args.split, encoding="utf-8-sig") as fh:
        items = [r for r in csv.DictReader(fh) if r["split"] == args.split_name]

    # 每类抽样：优先选该图 tile 预测不一致（信息量大）的
    by_cls = defaultdict(list)
    for it in items:
        by_cls[it["cls"]].append(it)
    chosen = []
    for c in CLASS_ORDER:
        g = by_cls.get(c, [])
        step = max(1, len(g) // max(args.per_class, 1))
        chosen += g[::step][:args.per_class]
    print(f"抽样 {len(chosen)} 个 tile（每类 {args.per_class}）")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx_rows, aopc_rows = [], []
    for it in chosen:
        p = os.path.join(args.data_root, "tiles", it["cls"], it["tile"])
        with Image.open(p) as im:
            im = im.convert("RGB").resize((args.img_size, args.img_size), Image.BICUBIC)
            arr0 = np.asarray(im).astype(np.float32) / 255.0
        a = (arr0 - arr0.mean()) / (arr0.std() + 1e-6)
        x = torch.from_numpy(a.transpose(2, 0, 1)[None].copy()).float().to(device)

        pred = int(model(x).argmax(1).item())
        cam, _ = hirescam(model, layer, x, pred)
        A = aopc(model, x, pred, cam, args.aopc_steps, device)
        aopc_m, aopc_p, gain = A["aopc_margin"], A["aopc_prob"], A["aopc_gain"]

        # 叠加图
        import matplotlib.cm as cm
        big = np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                         .resize((args.img_size, args.img_size), Image.BICUBIC)) / 255.0
        heat = cm.jet(big)[..., :3]
        overlay = np.clip(0.45 * arr0 + 0.55 * heat, 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.8))
        for axi, (img, ttl) in zip(ax, [(arr0, "SEM tile"),
                                        (heat, "HiResCAM"),
                                        (overlay, "overlay")]):
            axi.imshow(img); axi.set_title(ttl, fontsize=10); axi.axis("off")
        fig.suptitle(f"{it['tile']}  true={it['cls']}  pred={CLASS_ORDER[pred]}  "
                     f"AOPC_gain={gain:+.3f}", fontsize=10)
        fig.tight_layout()
        name = f"{os.path.splitext(it['tile'])[0]}_{backbone}.png"
        fig.savefig(os.path.join(args.out, "cam_overlays", name), dpi=170); plt.close(fig)

        peak = np.unravel_index(int(cam.argmax()), cam.shape)
        idx_rows.append({"tile": it["tile"], "source_image": it["source_image"],
                         "cls_true": it["cls"], "cls_pred": CLASS_ORDER[pred],
                         "correct": "T" if it["cls"] == CLASS_ORDER[pred] else "F",
                         "aopc_margin": round(aopc_m, 4), "aopc_margin_random": round(
                             A["aopc_margin_random"], 4), "aopc_gain": round(gain, 4),
                         "aopc_prob": round(aopc_p, 4),
                         "cam_peak_y": int(peak[0]), "cam_peak_x": int(peak[1]),
                         "overlay": name})
        aopc_rows.append({"source_image": it["source_image"], "cls_true": it["cls"],
                          "cls_pred": CLASS_ORDER[pred],
                          "correct": "T" if it["cls"] == CLASS_ORDER[pred] else "F",
                          "aopc_margin": round(aopc_m, 4), "aopc_margin_random": round(
                              A["aopc_margin_random"], 4), "aopc_gain": round(gain, 4),
                          "aopc_prob": round(aopc_p, 4)})
        print(f"  {it['tile']:<44} true={it['cls']:<7} pred={CLASS_ORDER[pred]:<7} "
              f"CAM={aopc_m:+.4f} rand={A['aopc_margin_random']:+.4f} gain={gain:+.4f}")

    with open(os.path.join(args.out, "e3_cam_index.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(idx_rows[0].keys()))
        w.writeheader(); w.writerows(idx_rows)
    with open(os.path.join(args.out, "e3_aopc.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(aopc_rows[0].keys()))
        w.writeheader(); w.writerows(aopc_rows)

    # AOPC 分布图：显著区删除 vs 随机删除（关键对照）
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    cams = [r["aopc_margin"] for r in aopc_rows]
    rands = [r["aopc_margin_random"] for r in aopc_rows]
    gains = [r["aopc_gain"] for r in aopc_rows]
    xpos = np.arange(2)
    ax.bar(xpos - 0.18, [np.mean(cams), np.mean(rands)], 0.36,
           yerr=[np.std(cams), np.std(rands)], capsize=5,
           color=["#D93025", "#9AA0A6"], label=["HiResCAM-ranked", "random order"])
    ax.set_xticks(xpos); ax.set_xticklabels(["margin drop (CAM)", "margin drop (random)"])
    ax.set_ylabel("logit margin drop")
    ax.axhline(0, color="k", lw=.8)
    for i, v in enumerate([np.mean(cams), np.mean(rands)]):
        ax.text(i, v + 0.004, f"{v:+.4f}", ha="center", fontsize=10)
    ax.set_title(f"AOPC with random control: gain = {np.mean(gains):+.4f} "
                 f"(>0 = saliency faithful)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = os.path.join(args.out, "fig_e3_aopc.png")
    fig.savefig(p, dpi=200); plt.close(fig)

    def _m(key):
        v = [r[key] for r in aopc_rows]
        return round(float(np.mean(v)), 4) if v else None

    summ = {"ckpt": args.ckpt, "backbone": backbone, "n_samples": len(aopc_rows),
            "aopc_margin_mean": _m("aopc_margin"),
            "aopc_margin_random_mean": _m("aopc_margin_random"),
            "aopc_gain_mean": _m("aopc_gain"),
            "aopc_prob_mean": _m("aopc_prob"),
            "verdict": ("saliency faithful" if (_m("aopc_gain") or 0) > 0.01 else
                        "no localized causal region (decision distributed)")}
    json.dump(summ, open(os.path.join(args.out, "e3_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nAOPC 显著区={summ['aopc_margin_mean']}  随机={summ['aopc_margin_random_mean']}  "
          f"增益={summ['aopc_gain_mean']}  判定：{summ['verdict']}")
    print(f"图 → {p}\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
