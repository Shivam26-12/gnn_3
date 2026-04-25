# DTNet M5: Advanced Graph Neural Network for Retail Demand Forecasting

This repository contains an advanced Graph Neural Network (GNN) architecture tailored specifically for the **M5 Walmart Forecasting Competition**. It models the entire M5 supply chain as a 30,490-node Item-Store graph with 426,860 supply-chain and substitution edges.

This codebase is fully optimized for **A100 GPU Training (80GB VRAM)** using `BFloat16` precision and massive tensor batching.

---

## 1. Architecture Overview

### The Graph
Instead of a simple hierarchical tree or isolated timeseries, we model demand explicitly using a flat graph structure:
- **Nodes (30,490):** Every unique product at every unique store (`item_store`).
- **Node Features (25):** 15 Dynamic features (lags, rolling stats, normalized prices, SNAP days) + 10 Static features (lifetime stats, state/category embeddings).
- **Edges (426,860):** 
  - `intra_dept`: Connects items on the same shelf to learn cannibalization and product substitution.
  - `cross_store`: Connects the same item across different stores in the same state to learn geographic demand shifts.
  - `inter_state`: Connects the same item across different states (CA ↔ TX ↔ WI) to learn Walmart's logistical inventory balancing.

### The Model
- **DTNetGNN:** A 3-Layer Graph Attention Network (GAT) with residual connections, `BatchNorm1d`, and `ELU` activations.
- **Loss Metric:** Differentiable RMSSE proxy during training; exact WRMSSE computation during validation to trigger early stopping.

---

## 2. Setup & Installation

### Environment Setup
You will need an environment with PyTorch and PyTorch Geometric installed. We highly recommend using `conda` or standard `pip` inside a virtual environment.

```bash
pip install -r requirements.txt
```
*(Make sure your PyTorch version matches your CUDA toolkit, e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu118`)*

### Data Placement
You must download the 3 main M5 dataset files from Kaggle:
1. `sales_train_evaluation.csv`
2. `calendar.csv`
3. `sell_prices.csv`

Place all three files inside the `data/raw/` directory at the root of the project:
```text
DTNet-main/
├── data/
│   └── raw/
│       ├── calendar.csv
│       ├── sales_train_evaluation.csv
│       └── sell_prices.csv
├── notebooks/
├── src/
├── run_pipeline.py
└── ...
```

---

## 3. Running the Pipeline

The entire process—from parsing the CSVs to training the Deep GAT on the A100—is automated in `run_pipeline.py`.

### End-to-End Run
To run the full pipeline (Graph Build → Dataset Slicing → A100 Training → Evaluation), simply run:
```bash
python run_pipeline.py
```

### Running Specific Steps
If you want to run specific parts of the pipeline:

**Step 1: Test Graph Construction**
Loads the CSVs and builds the 30k nodes and 426k edges. (Takes ~15 seconds).
```bash
python run_pipeline.py --step 1
```

**Step 2: Test Dataset Generation**
Slices the 1,941 days of history into 270 sliding windows and vectorizes the 25 features. (Takes ~55 seconds).
```bash
python run_pipeline.py --step 2
```

**Step 3: A100 Training (Heavy Compute)**
Pushes the graph to `cuda:0` and starts the GAT and MLP Baseline training loop. It will aggressively clear VRAM caches and use BFloat16 precision.
```bash
python run_pipeline.py --step 3
```

**Step 4: Final Evaluation**
Evaluates the best saved model (`results/dtnet_gnn_best.pt`) on the untouched test set and prints the final WRMSSE improvement percentage.
```bash
python run_pipeline.py --step 4
```

---

## 4. Notebooks & Simulation

The `/notebooks/` directory contains step-by-step interactive breakdowns of the architecture.

Additionally, the `/src/simulation/` folder contains a Mesa-based Digital Twin agent simulation. This allows you to synthetically trigger massive stockouts (`single_item_stockout`, `store_wide_disruption`, `category_recall`) across the graph to stress-test your trained GNN under extreme supply chain disruptions.
