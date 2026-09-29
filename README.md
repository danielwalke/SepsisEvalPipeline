# SepsisEvalPipeline: GraphFlow Framework for Sepsis Prediction & Interpretability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker Compose](https://img.shields.io/badge/docker--compose-v2+-blue.svg)](https://docs.docker.com/compose/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)](https://mlflow.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](https://modelcontextprotocol.io/)

**SepsisEvalPipeline / GraphFlow** is an end-to-end, reproducible, Docker-based graph learning workflow for early sepsis prediction from multi-center clinical laboratory time-series data (e.g., MIMIC-IV and SBC datasets), designed in adherence to FAIR (Findable, Accessible, Interoperable, Reusable) open science principles. 

It features time-decay temporal patient graph construction ($w = 1 - \Delta t_{\text{scaled}}$), memory-efficient mini-batch graph database storage (SQLite BLOB feature encoding & Neo4j), PyTorch Geometric Graph Attention Networks v2 (`GATv2`), the **GraphAware** spatial neighborhood feature aggregation framework, Model Context Protocol (MCP) AI integration, and an interactive Streamlit inference dashboard.

---

![GraphFlow Streamlit Dashboard Overview](docs/images/graphflow_dashboard_actual.png)

---

## Table of Contents

- [Key Features](#key-features)
- [Directory & Pipeline Overview](#directory--pipeline-overview)
- [Container Architecture & Docker Hub Deployment](#container-architecture--docker-hub-deployment)
  - [Published Docker Images (`dwalkeiti/...`)](#published-docker-images-dwalkeiti)
  - [Deployment Modes (Remote vs. Local Build)](#deployment-modes-remote-vs-local-build)
  - [Data Flow & Container Volume Bindings](#data-flow--container-volume-bindings)
  - [Standalone Step Execution (`docker run`)](#standalone-step-execution-docker-run)
- [Pipeline Execution Steps](#pipeline-execution-steps)
- [Runtime Profiling & Benchmarking](#runtime-profiling--benchmarking)
- [Environment Setup & Configuration (`.env`)](#environment-setup--configuration-env)
- [Interactive Dashboard Visualizations & Interpretability](#interactive-dashboard-visualizations--interpretability)
- [Model Context Protocol (MCP) & AI Integration](#model-context-protocol-mcp--ai-integration)
  - [Available MCP Tools](#available-mcp-tools)
  - [Starting the MCP Server](#starting-the-mcp-server)
  - [MCP Server JSON Configuration (`mcp.json`)](#mcp-server-json-configuration-mcpjson)
  - [Running the MCP Client](#running-the-mcp-client)
- [System Requirements & Setup](#system-requirements--setup)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Citation & References](#citation--references)

---

## Key Features

- **End-to-End Containerized Pipeline**: Fully modular architecture built on Docker containers for preprocessing, graph construction, database upload, model training, and explainable inference.
- **FAIR Principles & Open Science**: Decoupled, OS-independent Docker workflow ensuring Findability, Accessibility, Interoperability, and Reusability across institutions.
- **Dynamic Laboratory Panel Support**: Flexible evaluation across standard and composite lab panels: Complete Blood Count (`CBC`), Basic Metabolic Panel (`BMP`), Coagulation (`COAG`), combinations (`CBC_BMP`, `CBC_COAG`, `CBC_BMP_COAG`), C-Reactive Protein (`CRP`), and Creatinine (`CREATININE`).
- **Time-Decay Temporal Patient Graphs**: Constructs patient-centric graph representations where edge weights reflect normalized time differences ($w = 1 - \Delta t_{\text{scaled}}$) between laboratory observations.
- **Memory-Efficient Graph Storage & Mini-Batching**: High-performance SQLite BLOB node feature storage and indexed edge list querying for low-RAM mini-batch training on standard hardware.
- **Diverse Machine Learning & Graph Suite**:
  - **Baseline ML**: Logistic Regression, Random Forest, XGBoost.
  - **Graph Neural Networks**: PyTorch Geometric Graph Attention Networks v2 (`GATv2`) with dynamic attention and edge-weight support.
  - **GraphAware**: 1-hop spatial neighborhood feature aggregation paired with XGBoost for fast, scalable graph learning.
- **Explainable AI ($2N$ Aggregated SHAP Values)**: Computes aggregated local and global SHAP values ($\text{SHAP}_{\text{orig}} + \text{SHAP}_{\text{delta\_mean}}$) per feature to deliver clinically interpretable explanations with publication-ready summary visualizations.
- **Publication-Grade Runtime Profiling**: Dedicated benchmarking scripts (`experiment_logging/plot_runtime_vertical.py`) providing vertically stacked log-scale comparisons of hyperparameter tuning, model training, and inference latencies.
- **Reproducibility & Deterministic Optimization**: Fixed random seeds (`np.random.default_rng(seed)`) and proper `space_eval` indexing ensure fully deterministic Bayesian hyperparameter searches across runs.
- **Cross-Dataset Generalizability**: Multi-center validation pipeline evaluating model transferability across distinct hospital cohorts (e.g., MIMIC-IV and SBC internal/external hospital cohorts).
- **F₂ Score (β=2) Cutoff Optimization**: Pre-computed optimal $F_2$ score classification cutoffs tailored per laboratory panel.
- **Model Context Protocol (MCP) Server & Client**: Standardized FastMCP server interface paired with an OpenAI-compatible MCP Client for LLM agent integration.
- **Streamlit Interactive Dashboard**: Real-time sepsis risk assessment, calibrated risk scores, ROC/AUROC evaluation, and patient SHAP visualizations.

---

## Directory & Pipeline Overview

```
SepsisEvalPipeline/
├── 0_mimic_preprocess/         # Step 0: MIMIC-IV raw extraction & itemid lab mapping
├── 1_preprocess/               # Step 1: Lab data normalization, panel filtering & splitting
├── 2_baseline/                 # Step 2: Baseline ML models (LogReg, Random Forest, XGBoost)
├── 3_graph_construction/       # Step 3: Temporal patient graph construction with time decay
├── 4_db_upload/                # Step 4: Upload graph nodes & edges to SQLite / Neo4j
├── 5_gnn_training/             # Step 5: PyTorch Geometric GNN (GATv2) mini-batch training
├── 6_graphaware/               # Step 6: GraphAware 1-hop spatial neighborhood XGBoost & SHAP
├── 7_inference/                # Step 7: Streamlit dashboard app & optimal F₂ score cutoffs
├── 8_example_use_cases/        # Step 8: Jupyter notebooks & programmatic usage examples
├── experiment_logging/         # Publication runtime profiling & vertical benchmark plotting
├── figures/                    # Publication-grade figures (runtime benchmarks, global SHAP attributions)
├── mcp_server/                 # FastMCP Server providing RPC tools for pipeline & LLM access
├── mcp_client.py               # OpenAI-compatible MCP Client script for LLM agent execution
├── docker-compose.remote.yml   # Zero-build deployment via remote Docker Hub images (dwalkeiti)
├── docker-compose.yml          # Core Docker Compose orchestration (hybrid local-build / tagged)
├── docker-compose-mcp.remote.yml # Zero-build deployment for MCP + MLflow using remote images
├── docker-compose-mcp.yml      # Docker Compose config for full MCP + MLflow service stack
├── docker-compose-ram.yml      # Low-RAM memory optimized Docker Compose configuration
├── pipeline.sh                 # Sequential bash wrapper script for steps 2 to 6
├── config.ini                  # Global system configuration (paths, panels, hyperparameters)
└── .env                        # Local environment variables (LLM credentials, HOST_UID, HOST_GID)
```

---

## Container Architecture & Docker Hub Deployment

All computational components of the SepsisEvalPipeline are packaged into self-contained, reproducible, multi-platform Docker container images hosted publicly on [Docker Hub under `dwalkeiti`](https://hub.docker.com/u/dwalkeiti). New users and institutional collaborators can immediately deploy and run the entire pipeline without installing complex local Python environments, PyTorch versions, R compilers, or CUDA dependencies.

### Published Docker Images (`dwalkeiti/...`)

| Step | Container Image | Description | Base / Environment | Default Ports | Recommended Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `dwalkeiti/sepsisevalpipeline-mimic-preprocessor-and-sbc-extractor:latest`<br>`dwalkeiti/sepsisevalpipeline-0-mimic-preprocess:latest` | MIMIC-IV clinical cohort extraction & Steinbach criterion mapping | R 4.4.0 (`rocker/r-ver`) + data.table | — | CPU (8 GB RAM) |
| **1** | `dwalkeiti/sepsisevalpipeline-1-datapreprocess:latest` | Panel feature filtering, normalization, imputation, train/val/test splitting | Python 3.14-slim | — | CPU (8 GB RAM) |
| **2** | `dwalkeiti/sepsisevalpipeline-2-baseline:latest` | Classical baseline training: Logistic Regression, Random Forest, XGBoost | Python 3.14-slim + scikit-learn | — | CPU (8 GB RAM) |
| **3** | `dwalkeiti/sepsisevalpipeline-3-graph-construction:latest` | Temporal patient graph generation with exponential time-decay weights | Python 3.14-slim + PyTorch cu130 | — | CPU / GPU (16 GB RAM) |
| **4** | `dwalkeiti/sepsisevalpipeline-4-db-upload:latest` | High-throughput SQLite BLOB node feature encoding & indexed edge tables | Python 3.10-slim + sqlite3 | — | CPU (4 GB RAM) |
| **5** | `dwalkeiti/sepsisevalpipeline-5-gnn-training:latest` | PyTorch Geometric `GATv2` mini-batch training with attention edge weighting | PyTorch 2.2.2 CUDA 12.1 runtime | — | NVIDIA GPU (16GB VRAM, 32GB RAM) |
| **6** | `dwalkeiti/sepsisevalpipeline-6-graphaware:latest` | GraphAware 1-hop spatial neighborhood aggregation + XGBoost + $2N$ SHAP | Python 3.11-slim + PyTorch cu130 | — | NVIDIA GPU / CPU (16 GB RAM) |
| **7** | `dwalkeiti/sepsisevalpipeline-7-graphflow-inference:latest` | Streamlit interactive clinical inference dashboard & local SHAP explorer | Python 3.11-slim + Streamlit | `8501:8501` | CPU (8 GB RAM) |
| **MCP** | `dwalkeiti/sepsisevalpipeline-mcp-server:latest` | FastMCP server providing standardized RPC tools for AI LLM agents | Python 3.11-slim + FastMCP | Stdio / RPC | CPU (4 GB RAM) |
| **MLflow** | `ghcr.io/mlflow/mlflow:latest` | Centralized experiment tracking server with SQLite backend & artifact store | Python 3 | `5000:5000` | CPU (2 GB RAM) |

---

### Deployment Modes (Remote vs. Local Build)

#### Mode 1: Remote-First Deployment (Recommended for New Users)
Using `docker-compose.remote.yml`, you do not need to compile or build any Dockerfiles locally. Docker automatically pulls the verified pre-built images from Docker Hub:

```bash
# 1. Pull all pipeline container images
docker compose -f docker-compose.remote.yml pull

# 2. Run the complete pipeline end-to-end
docker compose -f docker-compose.remote.yml up

# Or run individual steps sequentially:
docker compose -f docker-compose.remote.yml up 1-datapreprocess
docker compose -f docker-compose.remote.yml up 2-baseline
docker compose -f docker-compose.remote.yml up 3-graph-construction
docker compose -f docker-compose.remote.yml up 4-db-upload
docker compose -f docker-compose.remote.yml up 5-gnn-training
docker compose -f docker-compose.remote.yml up 6-graphaware

# Launch the Streamlit Dashboard and MLflow Tracking Server in the background:
docker compose -f docker-compose.remote.yml up -d mlflow-server 7-graphflow-inference
```

#### Mode 2: Hybrid / Local Development (`docker-compose.yml`)
If you modify code, Dockerfiles, or pipeline dependencies, use `docker-compose.yml`. Each service specifies both `image: dwalkeiti/...` and a local `build:` context:

```bash
# Rebuild any modified local containers and start execution
docker compose up --build

# Pull remote images if not building locally
docker compose pull
```

#### Mode 3: Low-RAM Memory Profiling (`docker-compose-ram.yml`)
For resource-constrained workstations or memory profiling, `docker-compose-ram.yml` automatically mounts a background `docker:cli` tracker container that logs peak resident memory per container into `peak_ram.txt`:

```bash
docker compose -f docker-compose-ram.yml up
```

#### Mode 4: Remote MCP Server & AI Agent Stack (`docker-compose-mcp.remote.yml`)
To deploy MLflow, Streamlit, and the FastMCP Server using remote images for external AI clients:

```bash
docker compose -f docker-compose-mcp.remote.yml up -d
```

---

### Data Flow & Container Volume Bindings

The pipeline uses explicit host volume mounts to pass artifacts between container steps while preserving host file ownership through `${HOST_UID}` and `${HOST_GID}`:

```mermaid
flowchart TD
    subgraph S0["Step 0: MIMIC Preprocessing"]
        C0["dwalkeiti/...-mimic-preprocessor"]
    end
    subgraph S1["Step 1: Data Preprocess"]
        C1["dwalkeiti/...-1-datapreprocess"]
    end
    subgraph S2["Step 2: Baselines"]
        C2["dwalkeiti/...-2-baseline"]
    end
    subgraph S3["Step 3: Graph Construction"]
        C3["dwalkeiti/...-3-graph-construction"]
    end
    subgraph S4["Step 4: Database Upload"]
        C4["dwalkeiti/...-4-db-upload"]
    end
    subgraph S5["Step 5: GNN Training"]
        C5["dwalkeiti/...-5-gnn-training"]
    end
    subgraph S6["Step 6: GraphAware & SHAP"]
        C6["dwalkeiti/...-6-graphaware"]
    end
    subgraph S7["Step 7: Dashboard"]
        C7["dwalkeiti/...-7-graphflow-inference"]
    end

    RAW["./mimic (Raw MIMIC-IV)"] -->|"/app/input"| C0
    C0 -->|"/app/output"| DIR0["0_mimic_preprocess/preprocessed_file/"]
    DIR0 -->|"/app/input"| C1
    C1 -->|"/app/output"| DIR1["1_preprocess/data/preprocessed_data/"]
    DIR1 -->|"/app/input"| C2
    DIR1 -->|"/app/input"| C3
    C3 -->|"/app/output"| DIR3["3_graph_construction/data/"]
    DIR3 -->|"/app/csv_data"| C4
    C4 -->|"/app/db"| DB["4_db_upload/sqlite/sqlite_data/"]
    DB -->|"/app/db"| C5
    DB -->|"/app/db"| C6
    C2 -->|Log Metrics| ML["MLflow Server (Port 5000)"]
    C5 -->|Log Metrics| ML
    C6 -->|Log Metrics| ML
    C6 -->|Models & SHAP| DIR6["6_graphaware/models/"]
    DIR6 --> C7
    C7 --> UI["Web Browser (Port 8501)"]
```

---

### Standalone Step Execution (`docker run`)

Each container can also be executed independently via standard `docker run`. Below are concrete command templates:

#### Step 1: Preprocessing Standalone
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "${PWD}/0_mimic_preprocess/preprocessed_file:/app/input" \
  -v "${PWD}/0_mimic_preprocess/features:/app/features" \
  -v "${PWD}/0_mimic_preprocess/extdata:/app/extdata" \
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/output" \
  -v "${PWD}/config.ini:/app/config/config.ini:ro" \
  dwalkeiti/sepsisevalpipeline-1-datapreprocess:latest
```

#### Step 3: Graph Construction Standalone
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "${PWD}/1_preprocess/data/preprocessed_data:/app/input" \
  -v "${PWD}/3_graph_construction/data:/app/output" \
  -v "${PWD}/3_graph_construction/metrics:/app/metrics" \
  -v "${PWD}/config.ini:/app/config/config.ini:ro" \
  dwalkeiti/sepsisevalpipeline-3-graph-construction:latest
```

#### Step 4: Database Upload Standalone
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "${PWD}/3_graph_construction/data:/app/csv_data:ro" \
  -v "${PWD}/4_db_upload/sqlite/sqlite_data:/app/db" \
  -v "${PWD}/config.ini:/app/config/config.ini:ro" \
  -e CSV_DIR=/app/csv_data \
  -e DB_PATH=/app/db/mimic_sbc_graph.db \
  dwalkeiti/sepsisevalpipeline-4-db-upload:latest
```

#### Step 7: Streamlit Dashboard Standalone
```bash
docker run --rm -d \
  -p 8501:8501 \
  --name graphflow_inference_app \
  --user $(id -u):$(id -g) \
  -v "${PWD}:/app" \
  -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  dwalkeiti/sepsisevalpipeline-7-graphflow-inference:latest
```

---

## Pipeline Execution Steps

### 0. MIMIC-IV Preprocessing (`0_mimic_preprocess/`)
- **Prerequisite**: MIMIC-IV requires a free [PhysioNet credentialed-access account](https://physionet.org/content/mimiciv/) ([documentation](https://mimic.mit.edu/docs/iv/)). Download the `hosp` module and place its CSVs under `./mimic/hosp/` in the repo root (e.g. `./mimic/hosp/labevents.csv`, `./mimic/hosp/d_labitems.csv`, ...) - `docker-compose.yml` mounts `./mimic` as `/app/input`. If this data is missing, `docker compose up` (and `pipeline.sh`) will fail fast with a message pointing back here instead of running.
- `0_mimic_preprocess/extdata/icumap.csv` is a small curated lookup table tracked in the repo; `0_mimic_preprocess/extdata/d_labitems.csv` is raw MIMIC-IV content and is instead copied in automatically from `./mimic/hosp/d_labitems.csv` at container startup - no manual setup needed for either.
- Preprocesses MIMIC-IV clinical data according to Steinbach et al. criteria.
- Maps raw lab item IDs to standardized lab codes using `panel_name_to_feature_codes.py` (also run automatically inside the container).
- **Output**: Preprocessed dataset files under `0_mimic_preprocess/preprocessed_file/`.

### 1. Pre-processing (`1_preprocess/`)
- Cleans, standardizes, and splits MIMIC-IV and SBC laboratory data into train, validation, and test sets.
- Filters observations according to configured lab panels (`config.ini`).
- **Output**: Clean CSV files under `1_preprocess/data/preprocessed_data/`.

### 2. Machine Learning Baselines (`2_baseline/`)
- Trains baseline Logistic Regression, Random Forest, and XGBoost classifiers.
- Logs evaluation metrics to MLflow.
- **Output**: Trained models in `2_baseline/models/`.

### 3. Temporal Graph Construction (`3_graph_construction/`)
- Constructs directed patient-centric temporal graphs where edges link sequential laboratory measurements.
- Calculates exponential time-decay edge weights based on time elapsed between measurements ($w = 1 - \Delta t_{\text{scaled}}$).
- **Output**: Graph node and edge CSV files in `3_graph_construction/data/`.

### 4. Database Upload (`4_db_upload/`)
- Imports graph nodes, edges, and feature vectors into SQLite (`sqlite_data/mimic_sbc_graph.db` using BLOB feature encoding and indexed edge tables) or Neo4j database instances.
- **Output**: SQLite / Neo4j database files.

### 5. GNN Training (`5_gnn_training/`)
- Fetches mini-batches from the graph database and trains Graph Attention Networks v2 (`GATv2`) with edge weight support via PyTorch Geometric.
- Incorporates deterministic Bayesian hyperparameter search via `ModelTuning.py` with seeded random state (`default_rng(seed)`).
- **Output**: Trained model checkpoints in `5_gnn_training/checkpoints/` and MLflow metric logs.

### 6. GraphAware Training & Global Interpretability (`6_graphaware/`)
- Extracts 1-hop spatial neighborhood features using time-decay weighted mean aggregations.
- Deterministically tunes XGBoost hyperparameters using Hyperopt TPE with `np.random.default_rng(42)` and `space_eval()` mapping.
- Computes $2N$ aggregated and raw SHAP feature attributions on internal validation and multi-center test cohorts (`interpret.py`).
- Generates publication-grade global summary plots (`shap_aggregated_<cohort>.png`) with stripped technical prefixes (`Total: `, `f__`) and bold clinical labels.
- Automatically syncs publication plots to `figures/`.
- **Output**: Trained GraphAware models in `6_graphaware/models/`, optimal parameters in `6_graphaware/hyperparameters/`, and SHAP plots in `6_graphaware/figures/` and `figures/`.

### 7. Interactive Inference & Dashboard (`7_inference/`)
- Interactive Streamlit dashboard (`app.py`) providing real-time sepsis risk prediction, calibrated probability percentage, ROC/AUROC curves, and local SHAP explanation breakdowns.

---

## Runtime Profiling & Benchmarking

The pipeline includes a dedicated benchmarking and visualization suite in `experiment_logging/plot_runtime_vertical.py` to evaluate computational efficiency across standard machine learning baselines, deep Graph Attention Networks (`GATv2`), and spatial GraphAware models.

- **3-Panel Vertical Benchmark**:
  1. **Hyperparameter Tuning Runtime (seconds, $\log_{10}$ scale)**: Automated search across model parameter spaces.
  2. **Model Training Runtime (seconds, $\log_{10}$ scale)**: End-to-end model fitting on full hospital training cohorts.
  3. **Inference Latency (seconds, $\log_{10}$ scale)**: Single-batch evaluation speed on held-out test cohorts.
- **Publication Plot Generation**:
  ```bash
  python experiment_logging/plot_runtime_vertical.py
  ```
  Generates `figures/runtime_vertical_plot.png` (and syncs to `runtime_vertical_plot.png`) with annotated median timings and publication hatchings.

---

## Environment Setup & Configuration (`.env`)

To run the pipeline services and Docker containers smoothly, create a `.env` file in the root directory.

> [!IMPORTANT]
> **User Permissions Requirement**: You **MUST** define `HOST_UID` and `HOST_GID` in your `.env` file so that files created inside Docker containers match the user and group IDs of your host machine.

Example `.env` configuration:

```env
# Host User & Group ID (Required for Docker container volume permissions)
HOST_UID=1000
HOST_GID=1000

# OpenAI-Compatible LLM Credentials (Required for MCP Client)
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://llm.bi.denbi.de/v1"
OPENAI_MODEL="vllm/google/gemma-4-31B-it"

# Pipeline & App Parameters
APP_NAME=SepsisEvaluationPipeline
LOG_LEVEL=INFO
SEED=42
```

Populate `HOST_UID` and `HOST_GID` on Linux/macOS using:
```bash
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
```

---

## Interactive Dashboard Visualizations & Interpretability

The Streamlit inference application provides three dedicated analytical views:

### 1. Sepsis Prediction Probabilities & Risk Calibrated Table

![Sepsis Prediction Probabilities & Calibrated Risk Table](docs/images/graphflow_predictions_overview.png)

**Explanation**:
- **Cutoff-Calibrated Sepsis Risk (%)**: Calibrates the raw output probability $P(\text{Sepsis})$ relative to the active $F_2$ score cutoff threshold $c$.
  - At raw probability $P = c$, calibrated risk is defined as **50.0%**.
  - Above the cutoff ($P \ge c$), risk increases continuously from 50% to 100%.
  - Below the cutoff ($P < c$), risk decreases continuously from 50% down to 0%.
- **Interactive Row Selection**: Users can click directly on any patient row in the table to trigger local SHAP feature explanations.

---

### 2. Ground-Truth Performance Evaluation (ROC Curve, AUROC & Confusion Matrix)

![Ground-Truth Performance Evaluation & ROC Curve](docs/images/graphflow_roc_auroc_evaluation.png)

**Explanation**:
- **ROC Curve & AUROC Score**: Displays the Receiver Operating Characteristic curve comparing True Positive Rate (Sensitivity) against False Positive Rate ($1 - \text{Specificity}$). The overall area under the curve (AUROC) summarizes discriminative performance across all decision thresholds.
- **Optimal Cutoff Star Marker ($\star$)**: Identifies the optimal threshold maximizing the $F_2$ score ($\beta=2$) weighting Sensitivity (Recall) 4x over Precision:

  $$F_2 = \frac{5 \times \text{PPV} \times \text{Sensitivity}}{4 \times \text{PPV} + \text{Sensitivity}}$$

- **Annotated Confusion Matrix**: Heatmap detailing True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) at the active decision threshold, alongside Sensitivity, Specificity, PPV (Precision), and NPV metrics.

---

### 3. Local SHAP Explanation & GraphFlow Feature Attribution ($2N$ Decomposed Values)

![Local SHAP Explanation & GraphFlow Feature Attribution](docs/images/graphflow_shap_explanation.png)

**Explanation**:
- **Total Aggregated Local SHAP Bar Chart**: Shows the net directional impact of each lab feature on the sepsis risk score for a selected patient.
  - **Red bars ($\text{SHAP} > 0$)**: Features driving the prediction *towards* high Sepsis risk (e.g., elevated Age or abnormal MCV).
  - **Blue bars ($\text{SHAP} < 0$)**: Protective features driving the prediction *away* from Sepsis (e.g., normal WBC or HGB levels).
- **GraphFlow Attribution Breakdown (Original vs. $\Delta$ Mean)**: Decomposes the total SHAP attribution into two distinct physical components:
  1. **Original Feature SHAP** ($\mathbf{X}_{\text{orig}}$): Contribution of the patient's current static laboratory values.
  2. **Time-Based $\Delta$ Mean SHAP** ($\mathbf{X}_{\text{orig}} - \boldsymbol{\mu}_{\text{neighbors}}$): Contribution of the patient's temporal trend relative to their historical 1-hop spatial neighborhood.
- **Detailed Attribution Breakdown Table**: Quantifies the exact raw values, time-based delta mean values, individual SHAP values, and risk impact directions for clinical auditability.

---

### 4. Global Feature Contributions & Aggregated SHAP Beeswarm Plots

![Aggregated Global SHAP Summary Plot](figures/shap_aggregated_MIMIC_TEST.png)

**Explanation**:
- **Aggregated Attribution**: Sums the static measurement impact ($\text{SHAP}_{\text{orig}}$) with the dynamic temporal neighborhood trajectory ($\text{SHAP}_{\Delta\text{mean}}$) for each laboratory analyte across all patients in the evaluation cohort.
- **Biomarker Impact Ranking**: Features are sorted vertically by overall contribution magnitude, revealing key sepsis drivers (e.g., elevated `WBC`, advancing `Age`, low `Bicarbonate`, and electrolyte imbalances).
- **Directional Effects**: Color indicates feature value (red = high, blue = low). Horizontal position reveals whether the value increases ($\text{SHAP} > 0$) or decreases ($\text{SHAP} < 0$) predicted sepsis probability.

---

## Model Context Protocol (MCP) & AI Integration

The repository includes a **FastMCP Server** ([mcp_server/server.py](file:///home/daniel.walke/git/SepsisEvalPipeline/mcp_server/server.py)) and an **OpenAI-Compatible MCP Client** ([mcp_client.py](file:///home/daniel.walke/git/SepsisEvalPipeline/mcp_client.py)), allowing LLM agents (e.g. OpenAI GPT-4o, DeepSeek, Ollama, vLLM, Groq, OpenRouter) to programmatically inspect and control the pipeline.

### Available MCP Tools

| MCP Tool Name | Description |
| :--- | :--- |
| `list_pipeline_steps` | Returns all pipeline steps (Steps 0 to 6) and their docker-compose services. |
| `run_pipeline_step` | Trains/builds a panel through `docker compose up --build <service>` per step, from MIMIC extraction (Step 0) through the requested step (or `all_steps`, through Step 6). Automatically runs any earlier steps whose output doesn't exist yet for the panel, and skips steps already trained for it (set `force_retrain=True` to redo them). OS-independent - runs each step in its own container, not the host Python environment. |
| `get_mlflow_experiment_results` | Queries MLflow SQLite DB for past experiment metrics and hyperparameters. |
| `get_optimal_cutoffs` | Fetches pre-calculated optimal $F_2$ score ($\beta=2$) classification cutoffs. |
| `run_graphflow_inference` | Runs GraphFlow 1-hop spatial neighborhood inference on sample datasets. |
| `explain_patient_prediction` | Computes $2N$ aggregated SHAP values for a specific patient observation. |
| `get_dashboard_status` | Checks if the Streamlit inference dashboard is active on port 8501. |

### Starting the MCP Server

You can start the FastMCP Server in two ways depending on your execution preference:

#### Method 1: Local Virtual Environment (Stdio Mode)

To run the MCP server directly using your local Python environment:

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch the FastMCP Server (runs in stdio mode)
python mcp_server/server.py
```

#### Method 2: Containerized Execution (Docker Compose)

To launch the MCP server as part of the full stack (alongside MLflow and the Streamlit Dashboard):

```bash
# Launch full service stack including mcp-server container
docker-compose -f docker-compose-mcp.yml up -d

# Or launch only the mcp-server service:
docker-compose -f docker-compose-mcp.yml up -d mcp-server
```

### MCP Server JSON Configuration (`mcp.json`)

To connect external LLM applications (such as Claude Desktop, Cursor, Antigravity, VS Code, or custom AI agents) directly to the GraphFlow FastMCP server, copy one of the following JSON configuration blocks into your client's `mcp.json` or `claude_desktop_config.json` file:

#### Configuration A: Local Virtual Environment Stdio Mode

*(Recommended for Claude Desktop, Cursor, or local IDEs running directly on your host machine)*

```json
{
  "mcpServers": {
    "sepsis-eval-pipeline": {
      "command": "/home/daniel.walke/git/SepsisEvalPipeline/.venv/bin/python",
      "args": [
        "/home/daniel.walke/git/SepsisEvalPipeline/mcp_server/server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

> **Note**: Replace `/home/daniel.walke/git/SepsisEvalPipeline` with the absolute path to your local repository clone.

#### Configuration B: Docker Compose Execution Mode

*(Recommended when running the pipeline in containerized environments)*

```json
{
  "mcpServers": {
    "sepsis-eval-pipeline": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "mcp_pipeline_server",
        "python",
        "/app/mcp_server/server.py"
      ]
    }
  }
}
```

### Running the MCP Client

You can run [mcp_client.py](file:///home/daniel.walke/git/SepsisEvalPipeline/mcp_client.py) using environment variables defined in [.env](file:///home/daniel.walke/git/SepsisEvalPipeline/.env):

```bash
# Ensure .env contains OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL
.venv/bin/python mcp_client.py --prompt "Check dashboard status and explain patient row index 0 for MIMIC_CBC panel."
```

Or pass flags explicitly:
```bash
.venv/bin/python mcp_client.py \
  --api-key "your-api-key" \
  --base-url "https://llm.bi.denbi.de/v1" \
  --model "vllm/google/gemma-4-31B-it" \
  --prompt "Run all pipeline steps and explain performance metrics."
```

---


## System Requirements & Setup

### Requirements
- **OS**: Linux / macOS / WSL2
- **Python**: 3.10+
- **Docker & Docker Compose**: Recommended for containerized deployment
- **Hardware**: 16 GB+ RAM (32 GB recommended for GNN graph construction), NVIDIA GPU (optional, for GNN/GraphAware acceleration)

### Installation

1. **Clone the repository** (with its `6_graphaware/GraphAware` submodule - Step 6 fails with `ModuleNotFoundError: No module named 'GraphAware.EnsembleFramework'` if this is skipped):
   ```bash
   git clone --recurse-submodules https://github.com/danielwalke/SepsisEvalPipeline.git
   cd SepsisEvalPipeline
   ```
   If already cloned without `--recurse-submodules`, run `git submodule update --init --recursive` instead.

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment (`.env` and `config.ini`)**:
   - Update `config.ini` for data paths, selected panel names, and hyperparameters.
   - Set up `.env` with required `HOST_UID`, `HOST_GID`, and LLM credentials.

### Docker Compose Quickstart

#### Option A: Zero-Build Remote Quickstart (Docker Hub)
Launch the MLflow tracking UI, the Streamlit Inference Dashboard, and the FastMCP server immediately without building any images locally:

```bash
# Pull and start services in detached mode
docker compose -f docker-compose-mcp.remote.yml up -d
```

#### Option B: Local Build Quickstart
Build and launch using local Dockerfiles:

```bash
docker compose -f docker-compose-mcp.yml up -d
```

- **MLflow Tracking UI**: `http://localhost:5000`
- **GraphFlow Dashboard**: `http://localhost:8501`
- **FastMCP Server**: Running on container `mcp_pipeline_server`

---

## MLflow Experiment Tracking

All training runs (baseline models, GNNs, and GraphAware) automatically record hyperparameters and metrics into the backend MLflow database (`mlflow_data/mlflow.db`).

Start MLflow manually:
```bash
./start_ml_flow.sh
```

---

## Citation & References

### Citation Placeholder

If you use GraphFlow or this framework in your research, please cite our paper:

```bibtex
@article{walke2024graphflow,
  title={GraphFlow: End-to-end graph learning workflow exemplified for predicting sepsis},
  author={Walke, Daniel and Staritzbichler, Ren{\'e} and Kaiser, Thorsten and Saake, Gunter and Broneske, David and Heyer, Robert},
  journal={Under Review / In Preparation},
  year={2024}
}
```

### Key References

1. **Steinbach et al. (2024)**: *Applying Machine Learning to Blood Count Data Predicts Sepsis with ICU Admission.* Clinical Chemistry. [DOI: 10.1093/clinchem/hvae001](https://doi.org/10.1093/clinchem/hvae001)
2. **Walke et al. (2025)**: *Edges are all you need: Potential of medical time series analysis on complete blood count data with graph neural networks.* PLOS ONE 20(7): e0327636. [DOI: 10.1371/journal.pone.0327636](https://doi.org/10.1371/journal.pone.0327636)
3. **Walke et al. (2025)**: *GraphAware: Interpretable machine learning on graphs.* Preprint at Research Square. [DOI: 10.21203/rs.3.rs-7471432/v1](https://doi.org/10.21203/rs.3.rs-7471432/v1)
4. **Lundberg & Lee (2017)**: *A Unified Approach to Interpreting Model Predictions.* NIPS 2017. [DOI: 10.5555/3295222.3295230](https://doi.org/10.5555/3295222.3295230)