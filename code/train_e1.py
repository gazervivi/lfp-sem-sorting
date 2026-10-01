#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e1.py — E1 基准实验：多骨干 × 双协议（P1 tile 级 / P2 图像级）

设计原则（针对 v1.2 暴露的问题）：
  1) 协议由外部 split csv 决定，训练脚本不感知协议 —— 保证两种协议完全同一套代码，
     这是"协议落差"结论可成立的前提。
  2) cosine LR 调度 + warmup，固定随机种子，cudnn.deterministic，
     每次 run 产出可复现的 config.json。
  3) 20 层固定日志：训练/验证 loss、acc、macro-F1，逐 epoch 落 CSV。
  4) 推理输出 tile 级预测 CSV（含四类概率），供 eval_e2 做聚合与拒判。
  5) 10GB 显存的保护：--amp 默认开启，--bs 可配，--accum 梯度累积。

用法：
  python train_e1.py \
      --data-root ./data_v2 \
      --split ./data_v2/splits/protocol_imagelevel_seed42.csv \
      --backbone resnet50 --epochs 80 --bs 32 --lr 3e-4 \
      --seed 42 --out ./runs/e1_p2_resnet50_s42
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_0Vcu"]
CLS2IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
class TileDataset(torch.utils.data.Dataset):
    """按 split csv 读取 tile；path = {data_root}/tiles/{cls}/{tile}"""

    def __init__(self, items, tiles_root, img_size=224, train=False, grayscale_norm=True):
        self.items = items
        self.tiles_root = tiles_root
        self.img_size = img_size
        self.train = train
        self.grayscale_norm = grayscale_norm
        self._paths = [os.path.join(tiles_root, it["cls"], it["tile"]) for it in items]

    def __len__(self):
        return len(self.items)

    def _augment(self, im):
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            im = im.transpose(Image.ROTATE_90)
        return im

    def __getitem__(self, idx):
        it = self.items[idx]
        with Image.open(self._paths[idx]) as im:
            im = im.convert("RGB")
            if self.train:
                # 随机缩放裁剪：模拟不同视野尺度，抑制绝对尺度捷径
                s = random.uniform(0.75, 1.0)
                W, H = im.size
                cw, ch = int(W * s), int(H * s)
                x = random.randint(0, W - cw)
                y = random.randint(0, H - ch)
                im = im.crop((x, y, x + cw, y + ch))
            im = im.resize((self.img_size, self.img_size), Image.BICUBIC)
            arr = np.asarray(im).astype(np.float32) / 255.0

        if self.grayscale_norm:
            # 逐图标准化：抑制不同加速电压/会话造成的整体亮度-衬度漂移（SEM 域偏移）
            m, s = arr.mean(), arr.std() + 1e-6
            arr = (arr - m) / s
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD

        arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        y = CLS2IDX[it["cls"]]
        return x, y, idx


def read_split(path, tiles_root=None):
    items = []
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            items.append(row)
    return items


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
def build_model(backbone, num_classes=4, freeze_backbone=False, weights=None):
    import torchvision.models as tvm

    note = ""
    if backbone == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        feat = m.fc.in_features
        m.fc = nn.Linear(feat, num_classes)
    elif backbone == "swin_t":
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif backbone == "swin_s":
        m = tvm.swin_s(weights=tvm.Swin_S_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif backbone == "efficientnet_b7":
        m = tvm.efficientnet_b7(weights=tvm.EfficientNet_B7_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif backbone in ("cspdarknet_se", "cspdarknet53", "seresnet50"):
        # v1.2 的 CSPDarknet+SE 以 timm 提供的带 SE 的残差骨干作等价基线
        try:
            import timm
        except ImportError:
            raise SystemExit("该骨干需要 timm：pip install timm")
        name = {"cspdarknet_se": "seresnet50", "cspdarknet53": "cspdarknet53",
                "seresnet50": "seresnet50"}[backbone]
        m = timm.create_model(name, pretrained=True, num_classes=num_classes)
        note = f"timm:{name}（CSPDarknet+SE 的等价基线，非 v1.2 原始实现）"
    elif backbone.startswith("dinov2"):
        try:
            import timm
        except ImportError:
            raise SystemExit("DINOv2 需要 timm：pip install timm")
        # DINOv2 原生输入为 518（patch=14）；dynamic_img_size=True 允许插值到 224 等尺寸
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=num_classes, dynamic_img_size=True)
        note = "timm DINOv2 ViT-S/14 自监督权重 + 线性探测头（dynamic_img_size）"
        freeze_backbone = True
    else:
        raise SystemExit(f"未知骨干 {backbone}")

    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        note += f" | 加载域预训练权重 {os.path.basename(weights)} " \
                f"(missing {len(missing)} / unexpected {len(unexpected)})"

    if freeze_backbone:
        for n, p in m.named_parameters():
            if not any(k in n for k in ("head", "fc", "classifier")):
                p.requires_grad = False
    return m, note


# ----------------------------------------------------------------------------
# 指标与出图
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", labels=list(range(4)), zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0)
    per_class = {CLASS_ORDER[i]: {"precision": round(float(p[i]), 4),
                                  "recall": round(float(r[i]), 4),
                                  "f1": round(float(f[i]), 4),
                                  "support": int(s[i])} for i in range(4)}
    return acc, f1m, per_class


def plot_curves(hist, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(ep, [h["val_f1_macro"] for h in hist], label="val macro-F1", color="tab:green")
    ax[2].set_title("Macro F1"); ax[2].set_xlabel("epoch"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(os.path.basename(os.path.dirname(out_png)), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_cm(cm, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_ORDER, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (test)", fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------
# 训练 / 评估
# ----------------------------------------------------------------------------
def run_epoch(model, loader, crit, opt=None, scaler=None, accum=1, device="cuda"):
    train = opt is not None
    model.train(train)
    tot_loss, n = 0.0, 0
    ys, ps = [], []
    if train:
        opt.zero_grad(set_to_none=True)
    for step, (x, y, _i) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=(scaler is not None)):
                out = model(x)
                loss = crit(out, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss / accum).backward()
                else:
                    (loss / accum).backward()
                if (step + 1) % accum == 0:
                    if scaler is not None:
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
        tot_loss += loss.item() * y.size(0); n += y.size(0)
        ys.extend(y.detach().cpu().tolist())
        ps.extend(out.detach().argmax(1).cpu().tolist())
    if train and (step + 1) % accum != 0:   # 尾部残余梯度
        if scaler is not None:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
    acc, f1m, _pc = compute_metrics(ys, ps)
    return tot_loss / max(n, 1), acc, f1m, ys, ps


@torch.no_grad()
def predict(model, loader, device="cuda"):
    model.eval()
    rows = []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
        prob = torch.softmax(logits.float(), 1).cpu().numpy()
        for k in range(len(idx)):
            rows.append((int(idx[k]), int(y[k]), prob[k]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="域预训练权重 .pth（跨体系迁移臂）")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-grayscale-norm", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=25, help="val acc 无提升的早停耐心")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 随机性控制
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] 未检测到 GPU，将在 CPU 上运行（仅用于连通性验证）")

    items = read_split(args.split)
    tiles_root = os.path.join(args.data_root, "tiles")
    tr = [it for it in items if it["split"] == "train"]
    va = [it for it in items if it["split"] == "val"]
    te = [it for it in items if it["split"] == "test"]
    print(f"split: {os.path.basename(args.split)}  "
          f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    gnorm = not args.no_grayscale_norm
    ds_tr = TileDataset(tr, tiles_root, args.img_size, train=True, grayscale_norm=gnorm)
    ds_va = TileDataset(va, tiles_root, args.img_size, train=False, grayscale_norm=gnorm)
    ds_te = TileDataset(te, tiles_root, args.img_size, train=False, grayscale_norm=gnorm)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                                        num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                                        num_workers=args.workers, pin_memory=True)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.bs, shuffle=False,
                                        num_workers=args.workers, pin_memory=True)

    model, note = build_model(args.backbone, 4,
                              freeze_backbone=False, weights=args.weights)
    model = model.to(device)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"backbone={args.backbone}  可训练参数={n_train_p/1e6:.2f}M  {note}")

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    steps = max(len(dl_tr) // args.accum, 1) * args.epochs
    warm = max(len(dl_tr) // args.accum, 1) * args.warmup

    def lr_at(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(steps - warm, 1)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0))) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not args.no_amp))

    cfg = {**vars(args), "classes": CLASS_ORDER, "device": device,
           "n_train_p": n_train_p, "model_note": note,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "torch": torch.__version__, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    log_path = os.path.join(args.out, "training_log.txt")
    logf = open(log_path, "w", encoding="utf-8")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()

    log("=" * 74)
    log(f"E1 run | backbone={args.backbone} seed={args.seed} protocol={os.path.basename(args.split)}")
    log(f"gpu={cfg['gpu']} torch={cfg['torch']} amp={not args.no_amp} grayscale_norm={gnorm}")
    log("=" * 74)
    log(f"{'ep':>4} {'lr':>9} {'tr_loss':>9} {'tr_acc':>8} {'va_loss':>9} "
        f"{'va_acc':>8} {'va_f1':>8} {'sec':>6}")

    hist, best = [], {"acc": -1, "epoch": -1}
    best_path = os.path.join(args.out, "best.pt")
    bad = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        tl, ta, tf, _, _ = run_epoch(model, dl_tr, crit, opt, scaler, args.accum, device)
        with torch.no_grad():
            vl, vacc, vf1, _, _ = run_epoch(model, dl_va, crit, None, None, 1, device)
        sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        dt = time.time() - te0
        hist.append({"epoch": ep, "lr": cur_lr, "train_loss": tl, "train_acc": ta,
                     "val_loss": vl, "val_acc": vacc, "val_f1_macro": vf1, "sec": dt})
        log(f"{ep:>4} {cur_lr:>9.2e} {tl:>9.4f} {ta:>8.4f} {vl:>9.4f} "
            f"{vacc:>8.4f} {vf1:>8.4f} {dt:>6.1f}")
        if vacc > best["acc"]:
            best = {"acc": vacc, "epoch": ep, "f1": vf1, "val_loss": vl}
            torch.save({"model": model.state_dict(), "epoch": ep, "backbone": args.backbone,
                        "classes": CLASS_ORDER, "config": cfg}, best_path)
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == args.epochs:
            plot_curves(hist, os.path.join(args.out, "curves.png"))
        if bad >= args.patience and ep >= 30:
            log(f"[early stop] val acc 连续 {bad} epoch 无提升")
            break

    with open(os.path.join(args.out, "per_epoch_metrics.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)
    plot_curves(hist, os.path.join(args.out, "curves.png"))

    # ---- 用最优权重在 val / test 上出预测 ----
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    log(f"\n最优 epoch={best['epoch']}  val_acc={best['acc']:.4f}  val_f1={best['f1']:.4f}")

    summary = {"backbone": args.backbone, "seed": args.seed, "protocol": os.path.basename(args.split),
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "best_epoch": best["epoch"], "best_val_acc": best["acc"],
               "best_val_f1": best["f1"], "epochs_run": len(hist),
               "total_min": round((time.time() - t0) / 60, 2), "config": cfg}

    for name, ds, dl in (("val", ds_va, dl_va), ("test", ds_te, dl_te)):
        if len(ds) == 0:
            continue
        rows = predict(model, dl, device)
        out_rows = []
        for i, y, prob in rows:
            it = ds.items[i]
            p = int(prob.argmax())
            out_rows.append({
                "tile": it["tile"], "source_image": it["source_image"],
                "cls_true": CLASS_ORDER[y], "cls_pred": CLASS_ORDER[p],
                "confidence": round(float(prob[p]), 6), "correct": "T" if p == y else "F",
                "prob_A": round(float(prob[0]), 6), "prob_B": round(float(prob[1]), 6),
                "prob_C": round(float(prob[2]), 6), "prob_D": round(float(prob[3]), 6),
            })
        with open(os.path.join(args.out, f"predictions_{name}.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        ys = [r[1] for r in rows]; ps = [int(r[2].argmax()) for r in rows]
        acc, f1m, pc = compute_metrics(ys, ps)
        cm = confusion_matrix(ys, ps, labels=list(range(4)))
        plot_cm(cm, os.path.join(args.out, f"cm_{name}.png"),
                f"{args.backbone} seed{args.seed} {name} tile-level")
        summary[f"{name}_tile_acc"] = round(float(acc), 4)
        summary[f"{name}_tile_f1_macro"] = round(float(f1m), 4)
        summary[f"{name}_per_class"] = pc
        log(f"[{name} tile 级] acc={acc:.4f}  macro-F1={f1m:.4f}")
        for c in CLASS_ORDER:
            d = pc[c]
            log(f"    {c:<7} P={d['precision']:.3f} R={d['recall']:.3f} "
                f"F1={d['f1']:.3f} n={d['support']}")

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    log("=" * 74)
    log(f"完成，耗时 {summary['total_min']} 分钟 → {args.out}")
    logf.close()


if __name__ == "__main__":
    main()
