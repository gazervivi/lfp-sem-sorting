#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_results.py — 汇总 runs/ 下所有结果，直接打印论文可用表格"""
import glob
import json
import os
import sys
from collections import defaultdict

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/root/lfp/runs"
CLS = ["A_new", "B_2V", "C_0V", "D_0Vcu"]


def wilson(k, n, z=1.96):
    """Wilson 95% 置信区间（小样本比例区间，比正态近似更稳健）"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load():
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "metrics.json"))):
        d = json.load(open(f, encoding="utf-8"))
        d["_run"] = os.path.basename(os.path.dirname(f))
        out.append(d)
    return out


def proto_short(p):
    """严格区分各协议，避免 kfold/blocked 因协议名含 'imagelevel' 被误并入 P2"""
    if p.startswith("protocol_tilelevel"):
        return "P1"
    if p.startswith("protocol_imagelevel"):
        return "P2"
    if "kfold" in p:
        return "KF"
    if "blocked" in p:
        return "P3"
    return p


def main():
    rows = load()
    if not rows:
        print("无结果")
        return
    print("=" * 104)
    print(" E1 主表（tile 级指标）")
    print("=" * 104)
    print(f"{'proto':<6}{'backbone':<18}{'seed':>5}{'val_acc':>10}{'test_acc':>10}"
          f"{'test_F1':>9}{'best_ep':>9}{'ran_ep':>7}{'min':>7}")
    for d in sorted(rows, key=lambda x: (proto_short(x["protocol"]), x["backbone"], x["seed"])):
        mins = d.get("total_min")
        mins_s = f"{mins:>7.1f}" if isinstance(mins, (int, float)) else f"{'-':>7}"
        print(f"{proto_short(d['protocol']):<6}{d['backbone']:<18}{d['seed']:>5}"
              f"{d.get('best_val_acc', 0):>10.4f}{d.get('test_tile_acc', 0):>10.4f}"
              f"{d.get('test_tile_f1_macro', 0):>9.4f}{d['best_epoch']:>9}"
              f"{d['epochs_run']:>7}{mins_s}")

    print("\n" + "=" * 104)
    print(" 分类别指标（test，tile 级）")
    print("=" * 104)
    print(f"{'run':<30}{'acc':>8}   " + "".join(f"{c.split('_')[0]+' P/R':>12}" for c in CLS))
    for d in sorted(rows, key=lambda x: (proto_short(x["protocol"]), x["backbone"], x["seed"])):
        pc = d.get("test_per_class", {})
        cells = "".join(f"{pc[c]['precision']:.2f}/{pc[c]['recall']:.2f}" .rjust(12) for c in CLS)
        print(f"{d['_run']:<30}{d.get('test_tile_acc', 0):>8.4f}   {cells}")

    print("\n" + "=" * 104)
    print(" 泄漏落差（同骨干同种子：P1 − P2，tile 级 test acc）")
    print("=" * 104)
    idx = defaultdict(dict)
    for d in rows:
        idx[(d["backbone"], d["seed"])][proto_short(d["protocol"])] = d.get("test_tile_acc")
    for (b, s), v in sorted(idx.items()):
        if "P1" in v and "P2" in v:
            gap = (v["P1"] - v["P2"]) * 100
            print(f"  {b:<20} seed={s}   P1 {v['P1']*100:6.2f}%  →  P2 {v['P2']*100:6.2f}%"
                  f"  落差 {gap:+.2f} pp")

    # ---- 三层协议阶梯（P1 / P2 / P3-blocked）----
    print("\n" + "=" * 104)
    print(" 三层协议阶梯（同骨干：tile 级 test acc）")
    print("=" * 104)
    ladder = defaultdict(dict)
    for d in rows:
        ladder[d["backbone"]][proto_short(d["protocol"])] = d.get("test_tile_acc")
    print(f"  {'backbone':<20}{'P1 tile':>12}{'P2 image':>12}{'P3 blocked':>13}"
          f"{'P1-P2':>10}{'P2-P3':>10}")
    for b, v in sorted(ladder.items()):
        f = lambda k: (f"{v[k]*100:5.2f}%" if v.get(k) is not None else "  -  ")
        g12 = (f"{(v['P1']-v['P2'])*100:+.2f}" if v.get("P1") and v.get("P2") else "-")
        g23 = (f"{(v['P2']-v['P3'])*100:+.2f}" if v.get("P2") and v.get("P3") else "-")
        print(f"  {b:<20}{f('P1'):>12}{f('P2'):>12}{f('P3'):>13}{g12:>10}{g23:>10}")

    # 图像级（来自 predictions_test.csv 的多数投票）
    print("\n" + "=" * 104)
    print(" 图像级精度（整图多数投票）+ Wilson 95% 置信区间")
    print("=" * 104)
    import csv
    from collections import Counter
    for d in sorted(rows, key=lambda x: (proto_short(x["protocol"]), x["backbone"])):
        p = os.path.join(RUNS, d["_run"], "predictions_test.csv")
        if not os.path.exists(p):
            continue
        grp = defaultdict(list)
        with open(p, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                grp[r["source_image"]].append(r)
        ok, consist = 0, []
        for img, g in grp.items():
            gt = g[0]["cls_true"]
            votes = Counter(x["cls_pred"] for x in g)
            top, n = votes.most_common(1)[0]
            if len(votes) > 1 and n == votes.most_common(2)[1][1]:
                avg = {c: sum(float(x["prob_" + c[0]]) for x in g) / len(g) for c in CLS}
                top = max(avg, key=avg.get)
            ok += int(top == gt)
            consist.append(n / len(g))
        lo, hi = wilson(ok, len(grp))
        print(f"  {proto_short(d['protocol'])}/{d['backbone']:<18} seed={d['seed']}  "
              f"整图 {ok:>3}/{len(grp):<3} = {ok/len(grp)*100:5.1f}%  "
              f"[95%CI {lo*100:.1f}–{hi*100:.1f}]  平均 tile 一致率 {sum(consist)/len(consist):.3f}")
    # tile 级 CI
    print("\n  tile 级 Wilson CI（测试 tile 数）")
    for d in sorted(rows, key=lambda x: (proto_short(x["protocol"]), x["backbone"])):
        n = d.get("n_test", 0)
        acc = d.get("test_tile_acc")
        if not n or acc is None:
            continue
        k = round(acc * n)
        lo, hi = wilson(k, n)
        print(f"  {proto_short(d['protocol'])}/{d['backbone']:<18} seed={d['seed']}  "
              f"k/n = {k}/{n} = {acc*100:5.2f}%  [95%CI {lo*100:.2f}–{hi*100:.2f}]")


if __name__ == "__main__":
    main()
