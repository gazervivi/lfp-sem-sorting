#!/usr/bin/env python3
"""v3 figures: (a) P1-P2 tile-gap bootstrap distributions per backbone;
(b) P2 repeated-seed vs P3 juxtaposition; (c) oracle vs validation-only
rejection operating points on the risk-coverage curve."""
import csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/Users/nicolas/Downloads/train (2)/server_results/runs"
OUT = "/Users/nicolas/Downloads/train (2)/output/analysis_v3"
CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_cu"]
rng = np.random.default_rng(7)
B = 4000

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_pred(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source_image"]].append(r)
    imgs = []
    for img, g in sorted(by.items()):
        gt = g[0]["cls_true"]
        tc = np.array([r["correct"] == "T" for r in g])
        preds = [r["cls_pred"] for r in g]
        votes = collections.Counter(preds)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            avg = np.array([[float(r[f"prob_{c[0]}"]) for c in CLASS_ORDER] for r in g]).mean(axis=0)
            vp = CLASS_ORDER[int(np.argmax(avg))]
        else:
            vp = top[0][0]
        ap = np.array([[float(r[f"prob_{c[0]}"]) for c in CLASS_ORDER] for r in g]).mean(axis=0)
        imgs.append(dict(gt=gt, tile_correct=tc, vote_pred=vp,
                         reject_conf=top[0][1] / len(g) * float(ap.max())))
    return imgs


def tile_acc(s):
    return float(np.mean(np.concatenate([x["tile_correct"] for x in s])))


def img_acc(s):
    return float(np.mean([x["vote_pred"] == x["gt"] for x in s]))


fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

# ---- (a) P1-P2 tile-gap bootstrap distributions ----
a = ax[0]
backs = [("ResNet50", "e1_P1_resnet50_s42", "e1_P2_resnet50_s42"),
         ("EffNet-B0", "e1_P1_efficientnet_b0_s42", "e1_P2_efficientnet_b0_s42"),
         ("Swin-T", "e1_P1_swin_t_s42", "e1_P2_swin_t_s42"),
         ("DINOv2", "e1_P1_dinov2_s42", "e1_P2_dinov2_s42")]
cols = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98"]
for i, (nm, r1, r2) in enumerate(backs):
    p1 = load_pred(os.path.join(RUNS, r1, "predictions_test.csv"))
    p2 = load_pred(os.path.join(RUNS, r2, "predictions_test.csv"))
    n1, n2 = len(p1), len(p2)
    gaps = np.empty(B)
    for b in range(B):
        s1 = [p1[j] for j in rng.integers(0, n1, n1)]
        s2 = [p2[j] for j in rng.integers(0, n2, n2)]
        gaps[b] = tile_acc(s1) - tile_acc(s2)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    parts = a.violinplot([gaps * 100], positions=[i], widths=0.7, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(cols[i]); pc.set_alpha(0.55)
    a.plot([i - 0.25, i + 0.25], [lo * 100, lo * 100], color="k", lw=1)
    a.plot([i - 0.25, i + 0.25], [hi * 100, hi * 100], color="k", lw=1)
    a.plot([i, i], [lo * 100, hi * 100], color="k", lw=1)
    a.scatter([i], [gaps.mean() * 100], color="k", zorder=3, s=14)
a.axhline(0, color="grey", ls="--", lw=1)
a.set_xticks(range(4)); a.set_xticklabels([b[0] for b in backs], fontsize=8.5)
a.set_ylabel("P1 − P2 tile-accuracy gap (pp)")
a.set_title("(a) Image-level cluster bootstrap of the\nP1−P2 gap (B=4000; bars = 95% percentile CI)", fontsize=9)

# ---- (b) P2 two seeds vs P3 ----
b = ax[1]
pts = {"P2 seed 42": ("e1_P2_resnet50_s42", "#2471a3"),
       "P2 seed 43": ("e1_P2_resnet50_s43", "#2471a3"),
       "P3 block 5": ("blocked_resnet50", "#1e8449")}
xs, labels = [0, 1, 2.6], []
for x, (lab, (run, c)) in zip(xs, pts.items()):
    imgs = load_pred(os.path.join(RUNS, run, "predictions_test.csv"))
    n = len(imgs)
    boots = np.array([img_acc([imgs[j] for j in rng.integers(0, n, n)]) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pt = img_acc(imgs)
    b.errorbar([x], [pt * 100], yerr=[[(pt - lo) * 100], [(hi - pt) * 100]],
               fmt="o", color=c, capsize=4, ms=6, label=lab)
    labels.append(lab)
b.set_xticks(xs); b.set_xticklabels(["P2\nseed 42", "P2\nseed 43", "P3\nblock 5"], fontsize=8.5)
b.set_ylabel("image-level accuracy (%)")
b.set_ylim(88, 101)
b.set_title("(b) Two P2 realizations vs the single P3\nhold-out (ResNet50; bootstrap 95% CI)", fontsize=9)

# ---- (c) oracle vs validation-only threshold ----
c = ax[2]
val = load_pred(os.path.join(RUNS, "e1_P2_resnet50_s42", "predictions_val.csv"))
test = load_pred(os.path.join(RUNS, "e1_P2_resnet50_s42", "predictions_test.csv"))
ths = np.linspace(0.3, 1.0, 200)
for imgs, nm, col, ls in [(val, "validation (n=35)", "#e67e22", "-"),
                          (test, "test (n=36, oracle)", "#c0392b", "-")]:
    covs, accs = [], []
    for th in ths:
        keep = [x for x in imgs if x["reject_conf"] >= th]
        if not keep:
            continue
        covs.append(len(keep) / len(imgs))
        accs.append(np.mean([x["vote_pred"] == x["gt"] for x in keep]))
    c.plot(np.array(covs) * 100, np.array(accs) * 100, color=col, ls=ls, lw=1.8, label=nm)
c.scatter([100], [97.22], marker="s", s=45, color="#e67e22", zorder=5,
          label="val-selected threshold (0.61) →\ncoverage 100%, sel-acc 97.2%")
c.scatter([80.6], [100], marker="*", s=130, color="#c0392b", zorder=5,
          label="oracle threshold (0.92) →\ncoverage 80.6%, sel-acc 100%")
c.set_xlabel("coverage (%)"); c.set_ylabel("selective accuracy (%)")
c.set_title("(c) Rejection score: validation-only threshold\nselection fails on 35 error-free val images", fontsize=9)
c.legend(fontsize=7, loc="lower left", frameon=False)

fig.tight_layout()
p = os.path.join(OUT, "fig_v3_statistics.png")
fig.savefig(p, dpi=220)
print("saved", p)
