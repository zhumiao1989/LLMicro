# LLMicro

**Robust metagenomic profiling of complex microbial communities using large language models**

LLMicro is a parameter recommendation system for metagenomic classification tools based on large language models (LLMs) and retrieval-augmented generation (RAG), designed to improve the robustness of complex microbial community profiling.

## Features

- **RAG-Enhanced Parameter Recommendation**: Combines retrieval-augmented generation (RAG) with large language models to automatically recommend optimal parameter configurations based on sample features and external evidence
- **Three Classification Tool Support**: Kraken2, Centrifuge, PathSeq (with unified reference database)
- **Simulated Data Generation**: Generates three complexity levels of simulated metagenomic datasets (low/medium/high)
- **Comprehensive Evaluation**: Multi-level evaluation metrics at read-level and profile-level
- **Visualization**: Automatically generates publication-ready figures (Figures 1-4)
- **Knowledge Base Support**: Hierarchical document chunking and vector retrieval with multi-round evidence retrieval

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/LLMicro.git
cd LLMicro

# Create conda environment
conda env create -f environment.yml
conda activate llmicro

# Install Python dependencies
pip install -r requirements.txt

# Install RAG dependencies (optional, recommended)
pip install sentence-transformers faiss-cpu
```

## Quick Start

### 1. Generate Simulated Data

```bash
python src/simulate_data.py \
    --config config/simulation_config.yaml \
    --output data/simulated/ \
    --type all \
    --n-samples 20 \
    --seed 42
```

### 2. Build Knowledge Base Index (Optional)

```bash
# Place relevant literature and tool documentation in data/knowledge/
# Then build the vector index
python src/rag_knowledge_base.py --documents data/knowledge --index data/index
```

### 3. Run LLM Parameter Recommendation

```bash
# With RAG-enhanced recommendation
python src/llm_recommender.py \
    --config config/llm_config.yaml \
    --tool kraken2 \
    --input data/simulated/low_complexity \
    --output results/recommendations/kraken2_params.json \
    --use-rag
```

### 4. Run Classification Tools

```bash
# Kraken2
python src/run_kraken2.py \
    --input data/simulated/sample_001.fastq \
    --database data/reference/kraken2 \
    --output results/classifications/kraken2 \
    --confidence 0.1 \
    --min-hit-groups 2

# Centrifuge
python src/run_centrifuge.py \
    --input data/simulated/sample_001.fastq \
    --database data/reference/centrifuge \
    --output results/classifications/centrifuge \
    --k 3 \
    --min-hitlen 20

# PathSeq
python src/run_pathseq.py \
    --input data/simulated/sample_001.fastq \
    --reference data/reference/reference.fasta \
    --host-reference data/reference/hg38.fasta \
    --microbe-reference data/reference/microbes.fasta \
    --output results/classifications/pathseq \
    --min-clipped-read-length 35 \
    --host-min-identity 98
```

### 5. Evaluate Results

```bash
python src/evaluate.py \
    --results results/classifications/ \
    --ground-truth data/simulated/ \
    --tool kraken2 \
    --output results/metrics/kraken2_metrics.csv
```

### 6. Generate Visualizations

```bash
python src/visualize.py \
    --metrics results/metrics/all_metrics.csv \
    --output results/figures/
```

## Complete Pipeline

```bash
# Run the complete pipeline
bash scripts/run_pipeline.sh \
    --n-samples 20 \
    --seed 42 \
    --provider anthropic \
    --use-rag
```

## Project Structure

```
LLMicro/
├── README.md
├── LICENSE
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── config/
│   ├── llm_config.yaml          # LLM and RAG configuration
│   ├── tools_config.yaml        # Classification tool configuration
│   └── simulation_config.yaml   # Simulated data configuration
├── src/
│   ├── __init__.py
│   ├── llm_recommender.py       # LLM parameter recommendation core (with RAG support)
│   ├── rag_knowledge_base.py    # RAG knowledge base module
│   ├── simulate_data.py         # Simulated data generation
│   ├── run_kraken2.py           # Kraken2 runner script
│   ├── run_centrifuge.py        # Centrifuge runner script
│   ├── run_pathseq.py           # PathSeq runner script
│   ├── evaluate.py              # Results evaluation
│   ├── visualize.py             # Visualization
│   └── utils/
│       ├── __init__.py
│       ├── io.py                # I/O utilities
│       ├── metrics.py           # Evaluation metrics calculation
│       └── prompts.py           # LLM prompts (with RAG evidence support)
├── scripts/
│   ├── build_database.sh        # Reference database construction
│   └── run_pipeline.sh          # Complete pipeline script
├── data/
│   ├── knowledge/               # Knowledge base documents (literature, tool docs)
│   ├── index/                   # RAG vector index
│   ├── simulated/               # Simulated data
│   ├── mock/                    # Mock community data
│   └── reference/               # Reference database
├── results/
│   ├── figures/                 # Output figures
│   ├── tables/                  # Output tables
│   └── metrics/                 # Evaluation metrics
└── tests/
    ├── test_llm.py
    ├── test_simulation.py
    ├── test_evaluation.py
    └── test_rag.py
```

## LLMicro Framework Overview

LLMicro consists of four major components:

1. **Parameter Knowledge Acquisition and RAG-based Evidence Construction**
   - Document collection and preprocessing (tool documentation, methodological literature)
   - Hierarchical chunking with metadata preservation
   - Vector indexing and evidence retrieval

2. **Sample-aware Feature Extraction and Parameter Search Space Definition**
   - Extract sample features from raw sequencing data (sequencing depth, quality distribution, host contamination, community complexity)
   - Predefined native parameter space constraints

3. **Evidence-constrained LLM Recommendation and Hallucination Control**
   - Structured output constraints (only generate valid parameter names and in-range values)
   - Multi-round retrieval for implicit parameter evidence
   - Retain default values when evidence support is insufficient

4. **Resource-aware Optimization Strategy**
   - Incorporate computational resource constraints (runtime, memory) into recommendation process
   - Multi-objective optimization (analytical performance vs. computational cost)

## Parameter Recommendation Ranges

| Tool | Parameter | Range | Role |
|------|-----------|-------|------|
| Kraken2 | --confidence | 0.0-1.0 | Increase precision, control misclassification |
| | --minimum-hit-groups | 1-5 | Minimum independent k-mer hit groups required |
| Centrifuge | -k | 1-10 | Balance precision and recall |
| | --min-hitlen | 15-35 | Minimum length of exact seed hit |
| PathSeq | --min-clipped-read-length | 30-40 | Control minimum effective read length for alignment |
| | --host-min-identity | 80-100 | Increase host removal stringency |
| | --min-score-identity | 0.80-1.0 | Control minimum identity for microbial alignment |
| | --identity-margin | 0.0-0.1 | Control retention range for multiple candidate matches |

## Evaluation Metrics

### Read-level (Taxonomic Binning)
- **Precision**: Proportion of true positives among reads classified as positive
- **Recall**: Proportion of true positive reads correctly identified
- **F1-score**: Harmonic mean of Precision and Recall

### Profile-level (Taxonomic Profiling)
- **Profiling F1-score**: Consistency between detected and true species sets
- **L1-norm error**: Absolute deviation between predicted and true abundances
- **L2 distance**: Overall distance between predicted and true abundances
- **Bray-Curtis dissimilarity**: Community structure difference
- **False Positive taxa**: Number of false positives at different abundance thresholds (>0.001%, >0.01%, >0.1%)

## Key Findings

Based on evaluation results from v3 documentation:

1. **Low-complexity communities (10-30 species)**: LLMicro primarily improves result purity through more stringent parameter configurations, with significant Precision improvement
2. **Medium-complexity communities (50-150 species)**: LLMicro improves community composition recovery and abundance reconstruction, with significantly reduced L1/L2 errors
3. **High-complexity communities (200-400 species)**: LLMicro significantly reduces low-abundance false positives, especially at routine abundance thresholds
4. **Real noise conditions (Mock community)**: LLMicro maintains advantages in improving result purity, with stable benefits under real experimental perturbations
5. **Resource trade-offs**: LLMicro incurs some computational cost increase, smallest for Kraken2, most pronounced for PathSeq

