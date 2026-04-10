# LLMicro 使用指南

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [详细配置](#详细配置)
4. [命令行工具](#命令行工具)
5. [API 使用](#api 使用)
6. [结果解释](#结果解释)
7. [故障排除](#故障排除)

---

## 安装

### 系统要求

- Python 3.10+
- Linux/macOS (Windows 需要 WSL)
- 至少 16GB RAM (推荐 64GB)
- 100GB 可用磁盘空间

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/your-org/LLMicro.git
cd LLMicro
```

#### 2. 创建 Conda 环境（推荐）

```bash
conda env create -f environment.yml
conda activate llmicro
```

#### 3. 或使用 pip 安装

```bash
pip install -r requirements.txt
```

#### 4. 安装 LLM 依赖（可选）

```bash
# Anthropic (Claude)
pip install anthropic

# OpenAI (GPT)
pip install openai
```

#### 5. 设置 API 密钥

```bash
# 对于 Anthropic
export ANTHROPIC_API_KEY='your-api-key-here'

# 对于 OpenAI
export OPENAI_API_KEY='your-api-key-here'
```

#### 6. 安装生物信息学工具

```bash
# 使用 conda 安装
conda install -c bioconda kraken2 centrifuge

# PathSeq 需要 GATK4
conda install -c bioconda gatk4
```

---

## 快速开始

### 完整流程

```bash
# 运行完整流程
bash scripts/run_pipeline.sh \
    --n-samples 20 \
    --seed 42 \
    --provider anthropic
```

### 分步执行

#### 1. 生成模拟数据

```bash
python src/simulate_data.py \
    --config config/simulation_config.yaml \
    --output data/simulated \
    --type all \
    --n-samples 20 \
    --seed 42
```

#### 2. 运行 LLM 参数推荐

```bash
python src/llm_recommender.py \
    --config config/llm_config.yaml \
    --tool kraken2 \
    --input data/simulated/low_complexity \
    --output results/recommendations/kraken2_params.json
```

#### 3. 运行分类工具

```bash
# 使用默认参数
python src/run_kraken2.py \
    --input data/simulated/sample_001.fastq \
    --database data/reference/kraken2 \
    --output results/classifications/kraken2_default \
    --confidence 0.0 \
    --min-hit-groups 1

# 使用 LLM 推荐参数
python src/run_kraken2.py \
    --input data/simulated/sample_001.fastq \
    --database data/reference/kraken2 \
    --output results/classifications/kraken2_llm \
    --confidence 0.1 \
    --min-hit-groups 2
```

#### 4. 评估结果

```bash
python src/evaluate.py \
    --results results/classifications \
    --ground-truth data/simulated \
    --tool kraken2 \
    --output results/metrics/kraken2_metrics.csv
```

#### 5. 生成可视化

```bash
python src/visualize.py \
    --metrics results/metrics/all_metrics.csv \
    --output results/figures
```

---

## 详细配置

### LLM 配置 (config/llm_config.yaml)

```yaml
api:
  provider: anthropic  # 或 openai
  model: claude-sonnet-4-5-20250929
  anthropic_api_key: ${ANTHROPIC_API_KEY}

generation:
  temperature: 0.1  # 低温度确保一致性
  max_tokens: 2048

parameter_ranges:
  kraken2:
    confidence:
      min: 0.0
      max: 1.0
      default: 0.0
    minimum_hit_groups:
      min: 1
      max: 5
      default: 1
```

### 模拟数据配置 (config/simulation_config.yaml)

```yaml
low_complexity:
  n_samples: 20
  species_range:
    min: 10
    max: 30
  abundance_distribution: uniform

medium_complexity:
  n_samples: 20
  species_range:
    min: 50
    max: 150
  abundance_distribution: lognormal

high_complexity:
  n_samples: 20
  species_range:
    min: 200
    max: 400
  abundance_distribution: stepped
```

---

## 命令行工具

### llmicro-recommend

LLM 参数推荐工具。

```bash
llmicro-recommend \
    --config config/llm_config.yaml \
    --tool kraken2 \
    --input data/simulated/sample_001 \
    --output results/recommendations/params.json
```

**选项:**
- `--config`: LLM 配置文件路径
- `--tool`: 分类工具 (kraken2/centrifuge/pathseq)
- `--input`: 输入样本目录
- `--output`: 输出推荐参数文件
- `--provider`: LLM 提供商 (anthropic/openai)
- `--model`: 模型名称

### llmicro-simulate

模拟数据生成工具。

```bash
llmicro-simulate \
    --config config/simulation_config.yaml \
    --output data/simulated \
    --type all \
    --n-samples 20 \
    --seed 42
```

**选项:**
- `--config`: 模拟配置文件
- `--output`: 输出目录
- `--type`: 数据类型 (low/medium/high/mock/all)
- `--n-samples`: 每种类型的样本数
- `--seed`: 随机种子

### llmicro-evaluate

结果评估工具。

```bash
llmicro-evaluate \
    --results results/classifications \
    --ground-truth data/simulated \
    --tool kraken2 \
    --output results/metrics/metrics.csv
```

**选项:**
- `--results`: 分类结果目录
- `--ground-truth`: 真实值目录
- `--tool`: 分类工具名称
- `--output`: 输出指标文件

### llmicro-visualize

可视化工具。

```bash
llmicro-visualize \
    --metrics results/metrics/all_metrics.csv \
    --output results/figures
```

**选项:**
- `--metrics`: 输入指标 CSV 文件
- `--output`: 输出图形目录

---

## API 使用

### Python API

#### LLM 参数推荐

```python
from src.llm_recommender import LLMRecommender

# 初始化推荐器
recommender = LLMRecommender(
    config_path='config/llm_config.yaml',
    provider='anthropic'
)

# 样本特征
sample_features = {
    'sequencing_depth': 1000000,
    'mean_read_length': 150,
    'mean_quality': 35,
    'estimated_complexity': 3.5
}

# 获取推荐
rec = recommender.recommend(
    tool='kraken2',
    sample_features=sample_features
)

print(f"推荐参数：{rec.parameters}")
print(f"理由：{rec.reasoning}")
```

#### 数据模拟

```python
from src.simulate_data import DataSimulator

# 初始化模拟器
simulator = DataSimulator(
    config_path='config/simulation_config.yaml',
    random_seed=42
)

# 生成低复杂度样本
samples = simulator.generate_low_complexity_samples(
    n_samples=20,
    output_dir='data/simulated/low_complexity'
)
```

#### 结果评估

```python
from src.evaluate import Evaluator
import pandas as pd

# 初始化评估器
evaluator = Evaluator()

# 评估单个样本
result = evaluator.evaluate_sample(
    sample_id='sample_001',
    tool='kraken2',
    parameter_mode='llm',
    predictions=pred_df,
    ground_truth=truth_df,
    predicted_abundances=pred_abund,
    true_abundances=true_abund
)

print(f"Read-level F1: {result.read_level.f1}")
print(f"Profiling F1: {result.profile_level.profiling_f1}")
```

#### 可视化

```python
from src.visualize import Visualizer
import pandas as pd

# 加载指标
metrics = pd.read_csv('results/metrics/all_metrics.csv')

# 初始化可视化器
visualizer = Visualizer(output_dir='results/figures')

# 生成所有图形
visualizer.generate_all_figures(metrics)
```

---

## 结果解释

### 评估指标

#### Read-level 指标

| 指标 | 说明 | 取值范围 |
|------|------|----------|
| Precision | 被判定为阳性的 reads 中真实阳性的比例 | 0-1 |
| Recall | 真实阳性 reads 中被正确识别的比例 | 0-1 |
| F1-score | Precision 与 Recall 的调和均值 | 0-1 |

#### Profile-level 指标

| 指标 | 说明 | 取值范围 |
|------|------|----------|
| Profiling F1 | 检出物种集合与真实物种集合的一致性 | 0-1 |
| L1-norm error | 预测丰度与真实丰度的绝对偏离 | 0-∞ |
| L2 distance | 预测丰度与真实丰度的整体距离 | 0-∞ |

#### False Positive 指标

| 指标 | 说明 |
|------|------|
| FP > 0.001% | 丰度>0.001% 的假阳性物种数 |
| FP > 0.01% | 丰度>0.01% 的假阳性物种数 |
| FP > 0.1% | 丰度>0.1% 的假阳性物种数 |

### 显著性标记

- `***`: p < 0.001
- `**`: p < 0.01
- `*`: p < 0.05
- `.`: p < 0.1
- `ns`: 不显著 (p ≥ 0.05)

---

## 故障排除

### 常见问题

#### 1. LLM API 调用失败

**错误**: `anthropic.APIConnectionError`

**解决**:
- 检查 API 密钥是否正确设置
- 确认网络连接正常
- 检查 API 配额限制

#### 2. 内存不足

**错误**: `MemoryError` 或工具崩溃

**解决**:
- 减少并发线程数
- 降低模拟样本的复杂度
- 增加系统内存或使用 swap 空间

#### 3. 分类工具未找到

**错误**: `kraken2: command not found`

**解决**:
```bash
# 使用 conda 安装
conda install -c bioconda kraken2
```

#### 4. 参考数据库缺失

**错误**: `Database not found`

**解决**:
```bash
# 运行数据库构建脚本
bash scripts/build_database.sh
```

### 获取帮助

- GitHub Issues: https://github.com/your-org/LLMicro/issues
- 文档：https://github.com/your-org/LLMicro/docs
