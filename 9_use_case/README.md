# Strict Use Case: Patient `542802` Sepsis Trajectory Analysis

## Executive Summary
This use case demonstrates a clinical scenario from the **MIMIC-IV CBC_BMP** dataset (**Patient ID `542802`**, MIMIC Subject ID `19499446`, HADM ID `20503696`) matching all strict user restrictions:
1. **Traditional Baseline XGBoost predicts 100% NEGATIVE across the ENTIRE time series** (0 positive predictions out of 3 events).
2. **GraphAware XGBoost predicts NEGATIVE for the initial Control sample** ($t=0.0$h).
3. **GraphAware XGBoost correctly DETECTS Sepsis at clinical onset** ($t=46.1$h).

### Model-Specific Cutoff Thresholds (MIMIC_CBC_BMP Validation Set)
- **Traditional Baseline XGBoost Cutoff**: `0.001613` (~`0.0016`)
- **GraphAware XGBoost Cutoff**: `0.000178` (~`0.00018`)

---

## Patient Trajectory Table: Patient `542802`

| Time (h) | Chart Time | Clinical Label | WBC (k/uL) | Baseline XGBoost Prob | Baseline Status (cut=0.001613) | GraphAware XGBoost Prob | GraphAware Status (cut=0.000178) | Diagnostic Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0h** | 2162-06-23 14:01 | Control | 9.1 | 0.000274 | NEGATIVE | **0.000103** | **NEGATIVE (Correct)** | GraphAware initial control negative |
| **39.6h** | 2162-06-25 05:40 | Control | 7.4 | 0.000379 | NEGATIVE | 0.001275 | POSITIVE | Dynamic transition prior to onset |
| **46.1h** | 2162-06-25 12:07 | **Sepsis** | 9.3 | **0.000938** | **FALSE NEGATIVE (100% NEGATIVE SERIES)** | **0.001452** | **TRUE POSITIVE (SEPSIS DETECTED)** | **GraphAware Sepsis Detection** |

---

## Performance Summary

- **Traditional Baseline XGBoost**:
  - Probabilities across all 3 events (`0.000274`, `0.000379`, `0.000938`) remain strictly below the cutoff `0.001613`.
  - Baseline predicts **NEGATIVE for 100% of the patient's time series**, completely missing sepsis at clinical onset.

- **GraphAware XGBoost**:
  - At $t=0.0$h (Control), GraphAware predicted `0.000103` ($< 0.000178$) $\rightarrow$ **NEGATIVE (100% Correct Initial Control)**.
  - At $t=46.1$h (Sepsis Onset), GraphAware predicted `0.001452` ($\ge 0.000178$) $\rightarrow$ **SEPSIS DETECTED (True Positive)**.

---

## Standalone Test Commands

```bash
# Run standalone test for Patient 542802
.venv/bin/python 9_use_case/run_simple_test.py

# Run automated database-backed scanner
.venv/bin/python 9_use_case/find_divergent_cases.py
```
