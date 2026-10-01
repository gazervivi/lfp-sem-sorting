#!/usr/bin/env python3
"""v3 revision analyses (all local, from saved predictions/features):

A1. Image-level cluster bootstrap CIs for P1/P2/P3/KF (replaces tile-level Wilson
    non-overlap significance testing).
A2. P1 - P2 gap bootstrap CI (independent bootstrap; test sets differ).
A3. Validation-only rejection-threshold selection, then single application to test.
A4. Silhouette in the original 2048-D feature space (t-SNE kept as visualisation only).
A5. P3 spatial-block composition / difficulty heterogeneity.
A6. P2 two-seed comparison (n=2 repeated-split evidence).
"""
import csv, json, math, os, collections
import numpy as np

RUNS = "/Users/nicolas/Downloads/train (2)/server_results/runs"
OUT = "/Users/nicolas/Downloads/train (2)/output/analysis_v3"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261001)
B = 10000
CLASS_ORDER = ["A_new", "B_2V", "C_0V", "D_cu"]


def load_pred(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    by_img = collections.defaultdict(list)
    for r in rows:
        by_img[r["source_image"]].append(r)
    imgs = []
    for img, g in sorted(by_img.items()):
        gt = g[0]["cls_true"]
        tile_correct = np.array([r["correct"] == "T" for r in g], dtype=bool)
        preds = [r["cls_pred"] for r in g]
        votes = collections.Counter(preds)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            avg = [np.mean([float(r[f"prob_{c[0]}"]) for r in g]) for c in CLASS_ORDER]
            vote_pred = CLASS_ORDER[int(np.argmax(avg))]
        else:
            vote_pred = top[0][0]
        avg_probs = np.array([[float(r[f"prob_{c[0]}"]) for c in CLASS_ORDER] for r in g]).mean(axis=0)
        soft_pred = CLASS_ORDER[int(np.argmax(avg_probs))]
        consist = top[0][1] / len(g)
        reject_conf = consist * float(avg_probs.max())
        imgs.append(dict(img=img, gt=gt, n_tiles=len(g),
                         tile_correct=tile_correct,
                         vote_pred=vote_pred, soft_pred=soft_pred,
                         vote_consistency=consist,
                         prob_mean_max=float(avg_probs.max()),
                         reject_conf=reject_conf))
    return imgs


def metrics(sample):
    """sample: list of image dicts (a bootstrap resample)."""
    n = len(sample)
    img_acc = np.mean([s["vote_pred"] == s["gt"] for s in sample])
    tile_acc = np.mean(np.concatenate([s["tile_correct"] for s in sample]))
    # image-level per-class recall -> macro recall (balanced accuracy), C recall
    recs, c_rec = [], np.nan
    for c in CLASS_ORDER:
        grp = [s for s in sample if s["gt"] == c]
        if grp:
            r = np.mean([s["vote_pred"] == c for s in grp])
            recs.append(r)
            if c == "C_0V":
                c_rec = r
    return dict(img_acc=img_acc, tile_acc=tile_acc,
                bal_acc=np.mean(recs) if recs else np.nan, c_recall=c_rec)


def boot_ci(imgs, keys=("img_acc", "tile_acc", "bal_acc", "c_recall")):
    n = len(imgs)
    idx = rng.integers(0, n, size=(B, n))
    stats = {k: np.empty(B) for k in keys}
    # vectorised-ish loop (B x n small)
    for b in range(B):
        samp = [imgs[i] for i in idx[b]]
        m = metrics(samp)
        for k in keys:
            stats[k][b] = m[k]
    out = {}
    for k in keys:
        v = stats[k]
        v = v[~np.isnan(v)]
        out[k] = dict(lo=float(np.percentile(v, 2.5)),
                      hi=float(np.percentile(v, 97.5)),
                      mean=float(np.mean(v)))
    return out


def point(imgs):
    return metrics(imgs)


RESULTS = {}

# ---------- A1/A2: bootstrap per protocol/backbone ----------
PROTO = {
    "P1": {"resnet50": "e1_P1_resnet50_s42", "efficientnet_b0": "e1_P1_efficientnet_b0_s42",
           "swin_t": "e1_P1_swin_t_s42", "dinov2": "e1_P1_dinov2_s42"},
    "P2": {"resnet50": "e1_P2_resnet50_s42", "efficientnet_b0": "e1_P2_efficientnet_b0_s42",
           "swin_t": "e1_P2_swin_t_s42", "dinov2": "e1_P2_dinov2_s42"},
    "P3": {"resnet50": "blocked_resnet50", "efficientnet_b0": "blocked_efficientnet_b0"},
    "KF": {"resnet50": "kfold_merged_resnet50", "efficientnet_b0": "kfold_merged_efficientnet_b0"},
}
store = {}
for proto, backs in PROTO.items():
    for bb, run in backs.items():
        p = os.path.join(RUNS, run, "predictions_test.csv")
        if not os.path.exists(p):
            continue
        imgs = load_pred(p)
        store[(proto, bb)] = imgs
        pt = point(imgs)
        ci = boot_ci(imgs)
        RESULTS[f"boot::{proto}::{bb}"] = dict(
            n_images=len(imgs), point={k: round(v, 4) for k, v in pt.items()},
            ci95={k: dict(lo=round(v["lo"], 4), hi=round(v["hi"], 4)) for k, v in ci.items()})

# P1 - P2 gap (independent bootstrap, per backbone)
for bb in ["resnet50", "efficientnet_b0", "swin_t", "dinov2"]:
    a, b2 = store.get(("P1", bb)), store.get(("P2", bb))
    if not a or not b2:
        continue
    na, nb = len(a), len(b2)
    ia = rng.integers(0, na, size=(B, na))
    ib = rng.integers(0, nb, size=(B, nb))
    gaps_tile, gaps_img, gaps_c = np.empty(B), np.empty(B), np.empty(B)
    for k in range(B):
        sa = [a[i] for i in ia[k]]; sb = [b2[i] for i in ib[k]]
        ma, mb = metrics(sa), metrics(sb)
        gaps_tile[k] = ma["tile_acc"] - mb["tile_acc"]
        gaps_img[k] = ma["img_acc"] - mb["img_acc"]
        gaps_c[k] = ma["c_recall"] - mb["c_recall"] if not (np.isnan(ma["c_recall"]) or np.isnan(mb["c_recall"])) else np.nan
    def ci_of(v):
        v = v[~np.isnan(v)]
        return dict(mean=round(float(np.mean(v)), 4),
                    lo=round(float(np.percentile(v, 2.5)), 4),
                    hi=round(float(np.percentile(v, 97.5)), 4),
                    p_gt0=round(float(np.mean(v > 0)), 4))
    RESULTS[f"gap::P1-P2::{bb}"] = dict(tile_acc=ci_of(gaps_tile), img_acc=ci_of(gaps_img),
                                        c_recall=ci_of(gaps_c))

# ---------- A3: validation-only threshold selection ----------
def select_threshold_val(val_imgs, max_val_err=0.05):
    """maximize val coverage s.t. val selective error <= max_val_err."""
    scores = np.array([v["reject_conf"] for v in val_imgs])
    corr = np.array([v["vote_pred"] == v["gt"] for v in val_imgs])
    best = None
    for th in np.unique(np.round(scores, 4)):
        keep = scores >= th
        if keep.sum() == 0:
            continue
        err = 1 - corr[keep].mean()
        cov = keep.mean()
        if err <= max_val_err and (best is None or cov > best[1]):
            best = (float(th), float(cov), float(err))
    return best

for bb in ["resnet50", "efficientnet_b0"]:
    val = load_pred(os.path.join(RUNS, PROTO["P2"][bb], "predictions_val.csv"))
    test = store[("P2", bb)]
    sel = select_threshold_val(val)
    if sel is None:
        # fall back: minimal-error thresholds
        sel = (None, None, None)
    th, val_cov, val_err = sel
    out = dict(n_val=len(val), val_threshold=th, val_coverage=val_cov, val_selective_err=val_err)
    if th is not None:
        keep = [t for t in test if t["reject_conf"] >= th]
        rej = [t for t in test if t["reject_conf"] < th]
        out.update(test_coverage=round(len(keep) / len(test), 4),
                   test_selective_acc=round(float(np.mean([t["vote_pred"] == t["gt"] for t in keep])), 4) if keep else None,
                   n_covered=len(keep), n_rejected=len(rej),
                   rejected=[t["img"] for t in rej],
                   rejected_correct=[bool(t["vote_pred"] == t["gt"]) for t in rej])
    RESULTS[f"thresh::{bb}"] = out

# ---------- A4: silhouette in original feature space ----------
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d = np.load(os.path.join(RUNS, "e4_resnet50", "e4_features_resnet50.npz"), allow_pickle=True)
X, y = d["X"], d["y"]
sil = {}
sil["raw2048_euclidean"] = float(silhouette_score(X, y, metric="euclidean"))
sil["raw2048_cosine"] = float(silhouette_score(X, y, metric="cosine"))
Xz = StandardScaler().fit_transform(X)
sil["zscore2048_euclidean"] = float(silhouette_score(Xz, y, metric="euclidean"))
for nc in (20, 50):
    Xp = PCA(n_components=nc, random_state=0).fit_transform(Xz)
    sil[f"pca{nc}_euclidean"] = float(silhouette_score(Xp, y, metric="euclidean"))
RESULTS["silhouette"] = {k: round(v, 4) for k, v in sil.items()}

# ---------- A5: P3 block composition ----------
rows = list(csv.DictReader(open("/tmp/blocked_imagelevel.csv", encoding="utf-8-sig")))
info = {}
for r in rows:
    img = r["source_image"]
    if img not in info:
        info[img] = dict(cls=r["cls"], split=r["split"], x=float(r["x"]), y=float(r["y"]))
comp = collections.defaultdict(lambda: collections.Counter())
span = collections.defaultdict(lambda: [1e9, -1e9])
for v in info.values():
    comp[v["split"]][v["cls"]] += 1
    s = span[v["split"]]
    s[0] = min(s[0], v["x"]); s[1] = max(s[1], v["x"])
RESULTS["p3_blocks"] = {sp: dict(classes=dict(c), n=sum(c.values()),
                                 x_range=[round(span[sp][0], 2), round(span[sp][1], 2)])
                        for sp, c in comp.items()}

# ---------- A6: P2 two seeds ----------
for seed_run in ["e1_P2_resnet50_s42", "e1_P2_resnet50_s43"]:
    imgs = load_pred(os.path.join(RUNS, seed_run, "predictions_test.csv"))
    pt = point(imgs)
    RESULTS[f"p2_repeat::{seed_run}"] = dict(n_images=len(imgs),
                                             point={k: round(v, 4) for k, v in pt.items()})

json.dump(RESULTS, open(os.path.join(OUT, "analysis_v3.json"), "w"), indent=1, ensure_ascii=False)

# ---------- readable summary ----------
L = []
def w(s=""):
    L.append(s); print(s)

w("=" * 78)
w("A1. IMAGE-LEVEL CLUSTER BOOTSTRAP (B=%d, 95%% percentile CI)" % B)
w("=" * 78)
for proto in ["P1", "P2", "P3", "KF"]:
    for bb in PROTO[proto]:
        k = f"boot::{proto}::{bb}"
        if k not in RESULTS:
            continue
        r = RESULTS[k]
        p, c = r["point"], r["ci95"]
        w(f"{proto:3s} {bb:18s} n={r['n_images']:3d}  "
          f"imgAcc {p['img_acc']:.4f} [{c['img_acc']['lo']:.4f},{c['img_acc']['hi']:.4f}]  "
          f"tileAcc {p['tile_acc']:.4f} [{c['tile_acc']['lo']:.4f},{c['tile_acc']['hi']:.4f}]")
        w(f"{'':24s}balAcc {p['bal_acc']:.4f} [{c['bal_acc']['lo']:.4f},{c['bal_acc']['hi']:.4f}]  "
          f"C-recall {p['c_recall']:.4f} [{c['c_recall']['lo']:.4f},{c['c_recall']['hi']:.4f}]")
w()
w("=" * 78)
w("A2. P1 - P2 GAP (independent bootstrap; test sets differ)")
w("=" * 78)
for bb in ["resnet50", "efficientnet_b0", "swin_t", "dinov2"]:
    k = f"gap::P1-P2::{bb}"
    if k not in RESULTS:
        continue
    g = RESULTS[k]
    w(f"{bb:18s} tile-gap {g['tile_acc']['mean']:+.4f} [{g['tile_acc']['lo']:+.4f},{g['tile_acc']['hi']:+.4f}] P(>0)={g['tile_acc']['p_gt0']:.3f}")
    w(f"{'':18s} img-gap  {g['img_acc']['mean']:+.4f} [{g['img_acc']['lo']:+.4f},{g['img_acc']['hi']:+.4f}] P(>0)={g['img_acc']['p_gt0']:.3f}")
    w(f"{'':18s} Crec-gap {g['c_recall']['mean']:+.4f} [{g['c_recall']['lo']:+.4f},{g['c_recall']['hi']:+.4f}] P(>0)={g['c_recall']['p_gt0']:.3f}")
w()
w("=" * 78)
w("A3. VALIDATION-ONLY THRESHOLD SELECTION (max coverage s.t. val selective err <= 5%)")
w("=" * 78)
for bb in ["resnet50", "efficientnet_b0"]:
    t = RESULTS[f"thresh::{bb}"]
    w(f"{bb}: n_val={t['n_val']}  th={t['val_threshold']}  valCov={t['val_coverage']}  valErr={t['val_selective_err']}")
    if t.get("test_coverage") is not None:
        w(f"   -> test: coverage={t['test_coverage']} ({t['n_covered']}/{t['n_covered']+t['n_rejected']})  "
          f"selectiveAcc={t['test_selective_acc']}  rejected={t['rejected']} correct={t['rejected_correct']}")
w()
w("=" * 78)
w("A4. SILHOUETTE IN ORIGINAL FEATURE SPACE (image-level 2048-D, n=176)")
w("=" * 78)
for k, v in RESULTS["silhouette"].items():
    w(f"  {k:24s} {v:.4f}")
w("  (t-SNE 2D silhouette reported earlier: 0.6809 - visualisation only)")
w()
w("=" * 78)
w("A5. P3 SPATIAL BLOCK COMPOSITION")
w("=" * 78)
for sp, r in RESULTS["p3_blocks"].items():
    w(f"  {sp:6s} n={r['n']:3d} x_range={r['x_range']} classes={r['classes']}")
w()
w("=" * 78)
w("A6. P2 REPEATED SEEDS (n=2 only)")
w("=" * 78)
for k, r in RESULTS.items():
    if k.startswith("p2_repeat::"):
        p = r["point"]
        w(f"  {k.split('::')[1]:24s} imgAcc {p['img_acc']:.4f} tileAcc {p['tile_acc']:.4f} balAcc {p['bal_acc']:.4f} C-recall {p['c_recall']:.4f}")

open(os.path.join(OUT, "analysis_v3_summary.txt"), "w").write("\n".join(L))
print("\nSaved to", OUT)
