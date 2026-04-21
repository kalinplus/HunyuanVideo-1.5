# VBench 集成到 HunyuanVideo-1.5 — 任务卡 v2

## 背景

HunyuanVideo-1.5 当前的 `generate.py` 仅支持单条/少量视频生成（T2V/I2V）。
VBench 是视频生成领域的主流自动评测基准，涵盖 16 个维度（主体一致性、运动平滑性、色彩准确度等）共 946 条 prompt。
在 `/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/eval/` 中已有基于旧版 HunyuanVideo 的 VBench 完整实现，可作为接口参考。
本次任务是将 VBench 的采样和评测流程适配到 HunyuanVideo-1.5 的新 pipeline 接口上，使其能够一键完成「批量生成 → 指标计算 → 分数汇总」。

## 最终目标

在 HunyuanVideo-1.5 项目中新增 3 个脚本 + 1 个 prompt 数据文件，支持从 VBench prompt 列表批量生成视频、自动计算 16 维指标、汇总为 Quality/Semantic/Total 分数。

---

## 预实施分析

### 1. 目标复述

**目标**：在 `scripts/` 下新增 VBench 评测的完整工具链（采样 → 评测 → 汇总），复用现有 `HunyuanVideo_1_5_Pipeline` 接口，输出与 VBench 标准兼容的视频文件。

**边界**：
- 只做 T2V 任务的 VBench 评测（VBench 16 个维度均为 T2V）
- 不动 `generate.py` 的现有接口和逻辑
- 不引入新的 ML 依赖（`vbench` Python 库除外）
- 不实现分布式采样（先单 GPU 跑通，后续可扩展）

### 2. 风险点与歧义

| # | 风险 | 影响 | 应对 |
|---|------|------|------|
| R1 | **视频尺寸匹配**：VBench 标准 480×640 (H×W)，HunyuanVideo-1.5 的 bucket 机制可能无法精确匹配此尺寸 | 评测维度如 spatial_relationship 对尺寸敏感 | 已确认使用 `--resolution 480p --aspect_ratio 3:4`，480p Transformer 已下载在 ckpts 目录（与 720p 同目录，通过命名区分），先试不行再调 bucket 参数 |
| R2 | ~~帧数差异~~ | ~~65 帧可能不在模型最优区间~~ | **已确认使用 65 帧**，65=4×16+1 满足 VAE 4n+1 约束 |
| R3 | **VBench 视频命名匹配**：VBench 库通过文件名 `{prompt}-{seed}.mp4` 反查 prompt | 文件名中的特殊字符可能导致匹配失败 | 需对 prompt 做安全文件名处理（参考 TaylorSeer 实现） |
| R4 | ~~prompt rewrite 关闭~~ | ~~rewrite 会改变 VBench 原始 prompt~~ | **已确认关闭 prompt rewrite** |
| R5 | ~~GPU 内存~~ | ~~480p 65 帧 + 无 SR，单卡 24GB 应可运行~~ | **已确认单卡足够，无需 offloading** |

### 3. 最小改动方案

**只新增文件，不修改现有代码：**

```
scripts/
├── eval_vbench_sample.py    # 新增：批量采样脚本
├── eval_vbench_calc.py      # 新增：16 维指标计算
└── eval_vbench_tabulate.py  # 新增：分数归一化与汇总

hyvideo/datasets/
└── VBench_full_info.json    # 新增：946 条 VBench prompt 数据
```

核心思路：
- `eval_vbench_sample.py` 导入 `HunyuanVideo_1_5_Pipeline`，遍历 prompt 列表调用 `pipe()`，保存视频为 VBench 兼容命名
- `eval_vbench_calc.py` 和 `eval_vbench_tabulate.py` 从 TaylorSeer 实现中适配（改路径、改维度列表、去掉 TaylorSeer 特有逻辑）
- 零侵入：不动 `generate.py`、`train.py`、pipeline 类

### 4. 已确认事项

- **480p Transformer**：已下载在 ckpts 目录（与 720p 同目录），通过文件名区分
- **视频尺寸**：`--resolution 480p --aspect_ratio 3:4`（目标 480×640 H×W），先试不行再调
- **帧数**：固定 65 帧（满足 4n+1）
- **prompt rewrite**：关闭（`prompt_rewrite=False`）
- **SR**：关闭
- **offloading**：不需要，单卡足够

---

## 分步计划

### Step 1: 复制 VBench prompt 数据文件

- **做什么**：将 `VBench_full_info.json` 复制到 `hyvideo/datasets/` 下，并验证数据格式正确
- **产出物**：`hyvideo/datasets/VBench_full_info.json`
- **验收**：`python -c "import json; d=json.load(open('hyvideo/datasets/VBench_full_info.json')); print(len(d), list(d[0].keys()))"` 输出 `946 ['prompt_en', 'dimension']`

### Step 2: 编写批量采样脚本 `eval_vbench_sample.py`

- **做什么**：
  - argparse 接收参数：`--pretrained_model_root`, `--resolution 480p`, `--aspect_ratio 3:4`, `--video_length 65`, `--num_inference_steps 50`, `--seed 42`, `--output_dir`, `--index_start`, `--index_end`, `--num_videos_per_prompt 1`
  - 加载 `VBench_full_info.json`，按 `[index_start, index_end)` 分片
  - 调用 `HunyuanVideo_1_5_Pipeline.create_pipeline()` 创建 pipeline（480p Transformer，无 SR，无 offloading）
  - 遍历 prompt，逐条调用 `pipe(prompt=..., prompt_rewrite=False, seed=..., num_videos_per_prompt=..., output_type="np", enable_sr=False)`
  - 将 numpy 视频用 `imageio` 保存为 `{prompt_sanitized}-{seed_offset}.mp4` (24fps)
  - 进度条显示当前/总数，失败跳过并记录
- **产出物**：`scripts/eval_vbench_sample.py`
- **验收**：
  ```bash
  conda run -n hunyuan python scripts/eval_vbench_sample.py \
    --pretrained_model_root ./ckpts \
    --resolution 480p --aspect_ratio 3:4 \
    --video_length 65 --num_inference_steps 50 \
    --seed 42 --index_start 0 --index_end 2 \
    --output_dir ./vbench_test_output
  ls ./vbench_test_output/  # 应看到 2 个 .mp4 文件
  ```

### Step 3: 编写指标计算脚本 `eval_vbench_calc.py`

- **做什么**：
  - 从 TaylorSeer `calc_vbench.py` 适配，去掉 TaylorSeer 特有逻辑
  - argparse：`--videos_path`, `--output_dir`, `--full_info_path`, `--device`, `--start_dim`, `--end_dim`
  - 遍历 16 个维度，调用 `vbench.VBench.evaluate()` 计算每个维度分数
  - 每个 GPU 负责部分维度（通过 `--start_dim` / `--end_dim` 控制）
  - 输出 JSON 到 `{output_dir}/vbench/{dimension}_eval_results.json`
- **产出物**：`scripts/eval_vbench_calc.py`
- **验收**：
  ```bash
  conda run -n hunyuan python scripts/eval_vbench_calc.py \
    --videos_path ./vbench_test_output \
    --output_dir ./vbench_test_output \
    --full_info_path hyvideo/datasets/VBench_full_info.json \
    --device cuda:0 --start_dim 0 --end_dim 2
  ls ./vbench_test_output/vbench/  # 应看到 2 个 _eval_results.json
  ```

### Step 4: 编写分数汇总脚本 `eval_vbench_tabulate.py`

- **做什么**：
  - 从 TaylorSeer `tabulate_vbench_scores.py` 适配
  - argparse：`--score_dir`（指向 `vbench/` 子目录）
  - 读取所有 `{dimension}_eval_results.json`
  - 对每个维度做 min-max 归一化（使用 VBench 标准范围）
  - 计算 Quality Score (权重4) + Semantic Score (权重1) + Total
  - 输出 `all_results.json` 和 `scaled_results.json`
- **产出物**：`scripts/eval_vbench_tabulate.py`
- **验收**：
  ```bash
  conda run -n hunyuan python scripts/eval_vbench_tabulate.py \
    --score_dir ./vbench_test_output/vbench
  cat ./vbench_test_output/vbench/scaled_results.json  # 应包含 quality_score, semantic_score, total_score
  ```

---

## 非目标

- 不修改 `generate.py`、`train.py` 或任何现有 pipeline/model 代码
- 不实现分布式采样（torchrun 多卡并行，后续按需加）
- 不引入新 ML 依赖（`vbench` 库除外，需用户自行 `pip install vbench`）
- 不做 I2V 任务的 VBench 评测
- 不实现 TaylorSeer / DeepCache / TeaCache 等加速方案的评测对比

## 参考

- VBench 采样参考：`/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/sample_video_vbench.py`
- VBench 评测参考：`/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/eval/vbench/calc_vbench.py`
- VBench 汇总参考：`/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/eval/vbench/tabulate_vbench_scores.py`
- VBench prompt 数据：`/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/eval/VBench_full_info.json`
- HunyuanVideo-1.5 Pipeline 接口：`hyvideo/pipelines/hunyuan_video_pipeline.py`

## 自动化验收命令

- 运行环境: `conda activate hunyuan`
- 执行命令格式: `conda run -n hunyuan python ...`

### Step 1 验收
```bash
conda run -n hunyuan python -c "
import json
d = json.load(open('hyvideo/datasets/VBench_full_info.json'))
print(f'Total prompts: {len(d)}')
print(f'Keys: {list(d[0].keys())}')
print(f'Sample: {d[0][\"prompt_en\"][:80]}... | dim: {d[0][\"dimension\"]}')
"
```
预期输出: `Total prompts: 946`, `Keys: ['prompt_en', 'dimension']`

### Step 2 验收
```bash
conda run -n hunyuan python scripts/eval_vbench_sample.py \
  --pretrained_model_root ./ckpts \
  --resolution 480p --aspect_ratio 3:4 \
  --video_length 65 --num_inference_steps 50 \
  --seed 42 --index_start 0 --index_end 2 \
  --output_dir ./vbench_test_output && \
ls -la ./vbench_test_output/*.mp4 | wc -l
```
预期输出: `2` (2 个 mp4 文件)

### Step 3 验收
```bash
conda run -n hunyuan python scripts/eval_vbench_calc.py \
  --videos_path ./vbench_test_output \
  --output_dir ./vbench_test_output \
  --full_info_path hyvideo/datasets/VBench_full_info.json \
  --device cuda:0 --start_dim 0 --end_dim 2 && \
ls ./vbench_test_output/vbench/*_eval_results.json | wc -l
```
预期输出: `2` (2 个维度的结果文件)

### Step 4 验收
```bash
conda run -n hunyuan python scripts/eval_vbench_tabulate.py \
  --score_dir ./vbench_test_output/vbench && \
conda run -n hunyuan python -c "
import json
r = json.load(open('./vbench_test_output/vbench/scaled_results.json'))
for k in ['quality_score', 'semantic_score', 'total_score']:
    print(f'{k}: {r.get(k, \"MISSING\")}')
"
```
预期输出: 三个分数字段均有值

## 成功条件

- 所有步骤验收命令通过（exit code 0）
- diff 范围仅在 `scripts/eval_vbench_*.py` 和 `hyvideo/datasets/VBench_full_info.json`
- 生成的视频文件命名与 VBench 库期望的格式一致
- 16 维度指标均可正确计算（完整运行时）

## 错误处理约定

- 某步失败：先分析原因，给出修复方案，等确认后再修
- 连续两次失败：停下来，列出可能原因，不继续盲目重试
- 环境/依赖问题：报告具体报错，不自行修改环境配置
