# Use Case Analysis: Model-Specific Cutoff Adoption & Automated Divergent Patient Trajectory Discovery

## Executive Summary
This use case demonstrates how adopting **model-specific decision cutoff thresholds** derived from validation set G-Mean ROC optimization (rather than forcing a single cutoff across different architectures) reveals critical clinical divergence between **Traditional XGBoost** and **GraphAware XGBoost** on the **MIMIC-IV CBC_BMP** dataset.

### Model-Specific Cutoff Thresholds (MIMIC_CBC_BMP Validation Set)
- **Traditional Baseline XGBoost Cutoff**: `0.001613` (~`0.0016`)
- **GraphAware XGBoost Cutoff**: `0.000178` (~`0.00018`)

---

## Key Divergent Patient Case: Patient `166147`

- **Internal Patient ID**: `166147`
- **MIMIC Subject ID**: `12922837`
- **MIMIC HADM ID**: `28437422`
- **Dataset**: `CBC_BMP` (Complete Blood Count + Basic Metabolic Panel)

| Time (h) | Chart Time | Clinical Label | WBC (k/uL) | PLT (k/uL) | Baseline XGBoost Prob | Baseline Status (cut=0.001613) | GraphAware XGBoost Prob | GraphAware Status (cut=0.000178) | Clinical Impact |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0h** | 2172-01-04 00:02 | Control | 21.2 | 209 | 0.006586 | POSITIVE | 0.024850 | POSITIVE | Initial screening |
| **6.0h** | 2172-01-04 06:00 | Control | 19.1 | 196 | 0.004787 | POSITIVE | 0.002350 | POSITIVE | Ward transfer |
| **30.4h** | 2172-01-05 06:25 | Control | 17.3 | 203 | 0.005161 | POSITIVE | 0.000944 | POSITIVE | Dynamic monitoring |
| **52.3h** | 2172-01-06 04:20 | Control | 11.2 | 214 | 0.001347 | NEGATIVE | 0.000801 | POSITIVE | WBC normalization |
| **76.4h** | 2172-01-07 04:25 | **Sepsis** | 11.1 | 220 | **0.000599** | **FALSE NEGATIVE (MISSED)** | **0.003936** | **TRUE POSITIVE** | **GraphAware Sepsis Detection (>6.5x Score)** |

---

## Why Traditional XGBoost Failed & GraphAware Succeeded

1. **Static Tabular Blind Spot (Traditional XGBoost)**: At $t = 76.4$h, the patient's WBC count normalized to 11.1 k/uL (nearing normal reference range $\le 11.0$ k/uL). Because Traditional XGBoost evaluates each blood draw in static isolation based purely on tabular values, seeing WBC = 11.1 k/uL caused Traditional XGBoost's estimated risk to drop to **`0.000599`** (well below its model-specific cutoff `0.001613`), producing a dangerous **False Negative**.
2. **Temporal Neighborhood Awareness (GraphAware XGBoost)**: GraphAware evaluates temporal neighborhood feature aggregations (`original_features - mean_neighbors`). By tracking feature trajectory shifts across prior blood draws, GraphAware recognized the true sepsis onset at $t = 76.4$h, predicting **`0.003936` (POSITIVE)**—over **6.5-fold higher probability** than Traditional XGBoost.

---

## Automated Dataset Scanner Script (`find_divergent_cases.py`)

We created [`9_use_case/find_divergent_cases.py`](file:///home/daniel.walke/git/SepsisEvalPipeline/9_use_case/find_divergent_cases.py), an automated script that scans all sepsis patients across the entire test set using model-specific cutoffs and outputs divergent patient cases (Baseline Misses & Baseline False Alarms).

To run the scanner:
```bash
.venv/bin/python 9_use_case/find_divergent_cases.py
```

---

## Trajectory Visualization

![Patient 166147 Sepsis Journey](patient_166147_sepsis_journey.png)
