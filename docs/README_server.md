# 服务器执行手册（A466 / RTX 3080 10G）

> 数据源：`/Users/nicolas/Downloads/原始图像/2026.2.9-XH/`（177 张，4 类）
> 服务器路径：`/root/lfp/`
> 连接：`ssh lfp-gpu`（已配置密钥免密 + 别名）

---

## 0. 目录约定

```
/root/lfp/
├── venv/                    Python 环境（torch + timm + sklearn ...）
├── data/
│   ├── raw/2026.2.9-XH/     原始图像（只读，绝不改动）
│   └── v2/                  预处理产物（images / tiles / splits / manifest）
├── scripts/                 全部脚本
├── runs/                    每次训练一个子目录（config + log + ckpt + 预测 CSV）
├── docs/                    清单 CSV + 论文提纲
└── logs/                    运行日志
```

**铁律**：`data/raw/` 只读；所有划分以 CSV 落盘（可复查、可复现）；每次 run 必须留 `config.json`。

---

## 1. 环境体检（1 分钟）

```bash
cd /root/lfp && /root/lfp/venv/bin/python scripts/check_env.py
```

关注：CUDA 可用、GPU 是 3080 10G、显存占用、磁盘 ≥ 20G。

---

## 2. 预处理与划分（约 2–5 分钟 CPU）

```bash
cd /root/lfp
/root/lfp/venv/bin/python scripts/prepare_dataset.py \
    --src  /root/lfp/data/raw/2026.2.9-XH \
    --out  /root/lfp/data/v2 \
    --tile 256 --n-tiles 16 \
    --ratios 0.6,0.2,0.2 \
    --seeds 42,43,44
```

做的事：
1. md5 去重（剔除 C 类重复对）
2. D 类 2560×1920 → 1280×960 下采样（统一物理尺度，消除分辨率捷径）
3. 每图切 16 个 256×256 tile
4. 生成两套协议 × 3 个种子的划分 CSV + `splits/summary.json`

**划分比例说明**：两套协议统一用 `0.6/0.2/0.2`，保证测试集规模一致、tile 数可比。
> 刻意不用 8:1.5:0.5 —— 那样每类测试只有 2 张图，per-class 指标没有统计意义。

---

## 3. E1 快速通道（核心：协议落差）

```bash
cd /root/lfp
cp scripts/*.py scripts/*.sh .
EPOCHS=60 bash run_fast3h.sh
```

跑的内容（3080 上两条流并发）：

| 批次 | 内容 | 说明 |
|---|---|---|
| 批次 1 | ResNet50 × P1、EfficientNet-B0 × P1 | P1 = tile 级随机划分（文献做法，有泄漏） |
| 批次 2 | ResNet50 × P2、EfficientNet-B0 × P2 | P2 = 图像级划分（无泄漏） |
| 批次 3（可选） | ResNet50 × P1/P2 × seed43 | 稳健性；`WITH_SEED43=1` 开启 |

单 run 监控：

```bash
tail -f /root/lfp/logs/run_P2_resnet50_s42.log
nvidia-smi          # 看显存/利用率
```

**优先序**（时间不够时按此保结果）：批次 1 → 批次 2 → 汇总 → 批次 3。

---

## 4. E2 聚合决策 + 拒判（CPU，1 分钟）

```bash
cd /root/lfp
for r in runs/e1_P2_*/; do
  n=$(basename $r)
  /root/lfp/venv/bin/python scripts/eval_e2_aggregate.py \
      --pred $r/predictions_test.csv --out runs/e2_$n --tag $n
done
```

产出：三种决策规则对比、risk-coverage 曲线、每图明细（含误判样本清单）。

---

## 5. 汇总出图（CPU，1 分钟）

```bash
cd /root/lfp
/root/lfp/venv/bin/python scripts/make_figures.py --runs-dir runs --out runs/figures
```

产出：`table_master.md`、`fig_protocol_waterfall.png`（**论文核心图**）、`fig_backbone_compare.png`。

---

## 6. E4 特征空间 + 跨模型错误（GPU 3 分钟）

```bash
cd /root/lfp
/root/lfp/venv/bin/python scripts/eval_e4_tsne.py \
    --data-root data/v2 \
    --split data/v2/splits/protocol_imagelevel_seed42.csv \
    --ckpt runs/e1_P2_resnet50_s42/best.pt \
    --runs-dir runs --out runs/e4_resnet50
```

产出：t-SNE 嵌入图（按类别 / 按图内一致性着色）、跨模型共同错误清单。

---

## 7. 回收结果到本地

```bash
# 本地执行
rsync -a lfp-gpu:/root/lfp/runs/ "/Users/nicolas/Downloads/train (2)/server_runs/"
```

---

## 时间预算（3080 10G，60 epoch）

| 阶段 | 时长 |
|---|---|
| 依赖安装（一次性） | 5–10 min |
| 数据上传 379MB | ~10 min |
| 预处理 | 2–5 min |
| E1 批次 1（2 并发） | ~25–35 min |
| E1 批次 2（2 并发） | ~25–35 min |
| E2 + 汇总 + E4 | ~10 min |
| **合计** | **≈ 1.5–2 h** |

---

## 已知风险与对策

| 风险 | 对策 |
|---|---|
| 并发两 job 显存超 10G | 降 `--bs 24` 或改串行；日志出现 OOM 立即改 |
| Swin-T 单跑约 60–80 min | 3 小时窗口内不跑；等长窗口补 |
| 类间成像条件不一致（A=13kV / C,D=15kV / B 混 13-15kV） | 逐图标准化 + 论文 limitation 明确披露 |
| 每类仅 1 颗电池（类别与个体混淆） | limitation 必写；cell 级协议列为 future work |
| D 类分辨率 2× | 预处理已下采样统一，并在论文中说明 |
