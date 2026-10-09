# Optimizing SBERT for Long-Text Clustering

An experimental pipeline for evaluating **sentence-transformer embeddings on long documents**. The project compares direct document encoding against sentence-level and block-level aggregation, benchmarks long-context Transformer baselines, and evaluates the resulting embeddings using unsupervised clustering.

The implementation lives in [`Optimizing SBERT for Long-Text Clustering.ipynb`](Optimizing%20SBERT%20for%20Long-Text%20Clustering.ipynb). It includes comparisons against reference results recorded in the notebook, visualization, embedding caching, and optional scalability measurements.

## Overview

Standard sentence-embedding models have limited input lengths, so encoding long articles or reviews directly may truncate important information. This project investigates two alternatives that preserve information across longer documents:

- **Default:** Encode the document once, truncating tokens beyond the configured maximum sequence length.
- **Sentence-Level (SL):** Split the document into sentences, embed qualifying sentences independently, and average their vectors.
- **Block-Level (BL):** Split tokenized text into fixed-length blocks, mean-pool the tokens in each block, and average the resulting block representations.

**Longformer** and **BigBird** serve as additional long-sequence baselines.

## Experimental pipeline

```text
CSV datasets (text + cluster labels)
              |
              v
     Text preprocessing
              |
              v
 Embedding model + long-text method
   |       |        |           |
 Default   SL       BL    Longformer/BigBird
   |       |        |           |
   +-------+--------+-----------+
              |
              v
     L2-normalized embeddings
              |
              v
          UMAP (5D)
              |
         +----+----+
         |         |
      K-Means   K-Medoids
         |         |
         +----+----+
              |
      20 random seeds
              |
              v
    ACC*, ARI, NMI, Silhouette,
          weighted F1
              |
              v
       Tables + visualizations
```

*The notebook's `ACC` metric is **Rand Score**, as in the supplied publisher implementation. It is **not** ordinary classification accuracy.*

## Datasets

All inputs must be CSV files in a `Datasets/` directory. Each CSV must contain `text` and `cluster` columns.

| Dataset | Expected file | Records in the notebook | Clusters | Contents |
| --- | --- | ---: | ---: | --- |
| Amazon Reviews | `amazon_5000.csv` | 5,000 | 5 | Reviews grouped by rating |
| BBC News | `bbc_2250.csv` | 2,225 | 5 | News categories |
| Fake–Real News | `Fake_3000.csv` | 3,000 | 2 | Fake versus real news |

The project **does not download these datasets automatically**. Supply the three CSVs before running the notebook. Although its filename says `bbc_2250.csv`, the BBC dataset loaded during the recorded run contains **2,225 rows**.

## Models

| Model | Hugging Face identifier | Configured maximum length |
| --- | --- | ---: |
| BERT | `sentence-transformers/bert-base-nli-mean-tokens` | 128 |
| RoBERTa | `sentence-transformers/nli-roberta-base-v2` | 75 |
| DistilBERT | `sentence-transformers/distilbert-base-nli-mean-tokens` | 128 |
| ALBERT | `sentence-transformers/paraphrase-albert-small-v2` | 100 |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | 128 |
| Longformer | `allenai/longformer-base-4096` | 4,096 |
| BigBird | `google/bigbird-roberta-base` | 4,096 |

The five sentence-transformer models are tested with Default, SL, and BL. Longformer and BigBird are separately evaluated as long-sequence baselines.

## Setup

### 1. Requirements

- Python **3.11** (the notebook's recorded environment uses Python 3.11.9)
- Jupyter Notebook or JupyterLab
- PyTorch with CPU, CUDA, or compatible Intel XPU support
- Internet access on first run to download pretrained model weights
- The three dataset CSV files

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install jupyter ipykernel numpy pandas scipy scikit-learn scikit-learn-extra \
    sentence-transformers transformers umap-learn nltk psutil matplotlib seaborn openpyxl
```

For Intel Arc/XPU or CUDA builds, install the compatible PyTorch distribution for your hardware instead of assuming the generic `pip install torch` command enables acceleration. Some versions of `scikit-learn-extra` may require a compatible NumPy/scikit-learn/Python combination.

Download NLTK sentence-tokenization data if it is not already installed:

```bash
python -m nltk.downloader punkt punkt_tab
```

### 2. Project structure

```text
.
├── README.md
├── Optimizing SBERT for Long-Text Clustering.ipynb
├── Datasets/
│   ├── amazon_5000.csv
│   ├── bbc_2250.csv
│   └── Fake_3000.csv
└── generated_results/                # created by the notebook
    ├── embeddings/                   # cached .npy arrays
    ├── figures/                      # plots
    ├── tables/                       # CSV result tables
    └── SBERT_full_results.xlsx       # workbook
```

### 3. Run

Launch Jupyter **from the project root**, where the `Datasets/` folder is present:

```bash
jupyter notebook "Optimizing SBERT for Long-Text Clustering.ipynb"
```

Run the cells in order. The notebook verifies the environment, loads and preprocesses datasets, generates or loads embeddings, performs clustering, and writes results.

> **Path note:** The notebook uses `Path.cwd()` to locate `Datasets/`, but the missing-folder error message includes a machine-specific Windows path. You can run it from another computer or directory as long as the current working directory contains `Datasets/`.

## Experiment configuration

The following variables appear in the notebook's configuration cell:

```python
QUICK_TEST = False
QUICK_N = 100
RUN_FULL_EXPERIMENT = True
RUN_TABLE7 = True
RUN_VISUALIZATIONS = True
RUN_SCALABILITY_BENCHMARKS = False
REUSE_CACHED_EMBEDDINGS = True
PREFERRED_DEVICE = "xpu"
UMAP_SEED = 42
CLUSTER_SEEDS = list(range(20))
```

For an initial smoke test, set `QUICK_TEST = True`, and consider disabling optional comparisons and plots. The full experiment is compute-intensive: sentence-level encoding processes each sentence individually, and every clustering configuration is evaluated across 20 seeds. Cached embeddings under `generated_results/embeddings/` can be reused on subsequent runs.

Device selection prefers Intel **XPU** if configured and available, then CUDA, then CPU. The notebook also includes a CPU fallback for Longformer/BigBird operations that may be unsupported on a given XPU build.

## Clustering and evaluation

Embeddings are L2-normalized, then projected to **5 dimensions with UMAP** (`n_neighbors=70`, `min_dist=0.9`, Euclidean metric, seed 42). **K-Means** and **K-Medoids** use the dataset's specified number of clusters. Results are repeated with seeds `0` through `19`, and mean and population-standard-deviation summaries are saved.

| Metric | Interpretation |
| --- | --- |
| `ACC` | Rand Score: agreement over pairs of documents; **not** classification accuracy |
| `ARI` | Adjusted Rand Index: pairwise agreement corrected for chance |
| `NMI` | Normalized Mutual Information between predicted and reference clusters |
| `SIL` | Silhouette Score: cohesion and separation in the reduced embedding space |
| `F1` | Weighted F1 after mapping predicted cluster IDs to true labels with the Hungarian algorithm |

The labels are used for evaluation, not to train the unsupervised clustering models.

## Outputs

A completed run can create:

| File | Description |
| --- | --- |
| `generated_results/tables/clustering_results_mean_and_std.csv` | Aggregated metrics per dataset, algorithm, model, and method |
| `generated_results/tables/clustering_results_all_20_seeds.csv` | Individual results for all 20 clustering seeds |
| `generated_results/tables/paper_vs_local_clustering.csv` | Local metric values compared with reference values embedded in the notebook |
| `generated_results/tables/table7_cosine_similarity_local.csv` | Within-class cosine similarity for Fake and True news |
| `generated_results/tables/table7_paper_vs_local.csv` | Local/reference cosine-similarity comparison |
| `generated_results/tables/embedding_runtime_memory_throughput.csv` | Embedding runtime, process memory, and throughput measurements |
| `generated_results/SBERT_full_results.xlsx` | Consolidated result workbook |
| `generated_results/figures/fig8_acc_comparison_kmeans.png` | Model/method comparison for K-Means Rand Score |
| `generated_results/figures/fig9_bert_umap_2d.png` | Two-dimensional UMAP views of BERT embeddings |
| `generated_results/figures/fig10_model_correlation_heatmap.png` | Pearson correlations across model performance vectors |
| `generated_results/embeddings/*.npy` | Reusable normalized embedding arrays |

The optional scalability benchmark measures multiple sample sizes and can be enabled with `RUN_SCALABILITY_BENCHMARKS = True`. Its runtime and memory statistics are hardware-dependent; cached embedding lookup times in the main experiment are not equivalent to fresh-model inference times.

## Interpretation and reproducibility notes

- **Long-text aggregation is model-dependent:** Default, SL, and BL must be compared on the same dataset and clustering algorithm; one technique should not be assumed best universally.
- **Reference results are embedded in the notebook:** Local-versus-paper CSVs calculate differences, but differences alone do not establish exact experimental reproduction.
- **UMAP affects clustering:** The clustering algorithms operate on 5D UMAP features, not directly on the original embedding vectors.
- **Silhouette is internal:** Higher Silhouette does not necessarily imply better agreement with the dataset's reference labels.
- **BL implementation follows the notebook's reference code:** It manually inserts token IDs `101` and `102`; those IDs should not be assumed portable to every tokenizer without verification.
- **Results depend on the environment:** Package versions, model revisions, hardware, stochastic operations, and dataset preparation can affect output.

## Technologies

Python · PyTorch · Hugging Face Transformers · Sentence Transformers · NLTK · UMAP · scikit-learn · scikit-learn-extra · pandas · NumPy · Matplotlib · Jupyter

## Project scope

This repository is an **experimental evaluation notebook**, rather than a deployed clustering API or an SBERT fine-tuning framework. The emphasis is on alternative document-embedding strategies, unsupervised clustering quality, repeatable comparisons, and visualization.
