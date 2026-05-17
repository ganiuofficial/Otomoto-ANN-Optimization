# BAN6440 Module 6 — Otomoto Marketing Segmentation Model Optimisation
## ANN Optimisation: SGD vs Adam vs RMSProp | Nexford University | May 2026
**Author:** Ganiu Olalekan Mustapha

---

## Project Overview

This project optimises an Artificial Neural Network (ANN) for customer churn
prediction at Otomoto, using the Teleconnect dataset (7,043 customers). Three
optimisation algorithms are compared — SGD (baseline), Adam, and RMSProp —
on an identical ANN architecture to isolate the effect of the optimiser alone.

**Key results (actual code output):**

| Metric            | SGD (Baseline) | Adam   | RMSProp (Winner) |
|-------------------|---------------|--------|------------------|
| Accuracy          | 79.06%        | 79.21% | 79.56%           |
| Precision (Churn) | 61.93%        | 61.74% | 62.43%           |
| Recall (Churn)    | 54.81%        | 56.95% | 57.75%  <- KEY   |
| F1-Score (Churn)  | 58.16%        | 59.25% | 60.00%           |
| AUC-ROC           | 83.82%        | 83.78% | 83.81%           |
| Best val_loss     | 0.5300        | 0.4788 | 0.4826           |
| Epochs            | 100           | 60     | 51               |
| Training time     | 86.8s         | 55.2s  | 43.0s            |

**Winner: RMSProp** — highest F1 (60.00%) and recall (57.75%), fastest
convergence, most efficient. Recall is the primary marketing metric because
missing a churner (false negative) costs the full customer lifetime value,
while a false positive costs only the value of an unnecessary retention offer.

---

## Project Structure

```
module6_otomoto/
|
|-- otomoto_ann_optimisation.py      <- Main Python application
|-- README.md                        <- This file
|-- BAN6440_Module6_Report.docx      <- Full written report
|-- teleconnect.csv                  <- Dataset (provided by assignment)
|
|-- otomoto_outputs/                 <- Created automatically on run
    |-- 00_data_overview.png         <- EDA: churn distribution + monthly charges
    |-- 01_training_curves.png       <- Loss & accuracy curves for all 3 optimisers
    |-- 02_metrics_comparison.png    <- Grouped bar chart: all metrics compared
    |-- 03_roc_curves.png            <- Overlaid ROC curves for all 3 optimisers
    |-- 04_confusion_matrices.png    <- Confusion matrix per optimiser
    |-- 05_summary_table.png         <- Full summary table (green = best per metric)
    |-- metrics.json                 <- All metrics saved as JSON
```

---

## Dataset

**Name:** Teleconnect Customer Churn Dataset
**Provided by:** BAN6440 Module 6 assignment
**Records:** 7,043 customers | 21 raw columns | 30 encoded features
**Target:** Churn (Yes=1 / No=0) | Class balance: 26.5% churn / 73.5% no-churn
**Missing values:** 11 whitespace entries in TotalCharges — imputed with median

---

## Requirements

### Python Version
Python 3.9 or higher recommended.

### Install Dependencies

Open VS Code integrated terminal (Ctrl + `) and run:

    pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

Note for Windows users — if TensorFlow fails:

    pip install tensorflow-cpu

---

## How to Run in VS Code

### Step 1 — Open the project folder
    File -> Open Folder -> select the module6_otomoto folder

### Step 2 — Ensure teleconnect.csv is in the same folder as the .py file
The code reads the dataset with:
    df = pd.read_csv('teleconnect.csv')
The CSV must be in the working directory when you run the script.

### Step 3 — Open integrated terminal
    Terminal -> New Terminal   (or Ctrl + `)

### Step 4 — Run the application
    python otomoto_ann_optimisation.py

### Step 5 — Expected terminal output (condensed)

    =================================================================
      BAN6440 Module 6 | ANN Optimisation | Otomoto Marketing
      Dataset : teleconnect.csv (7,043 customers)
      Optimisers : SGD (Baseline) -> Adam -> RMSProp
    =================================================================

    STEP 1: DATA LOADING
       Shape            : (7043, 21)
       Churn=Yes        : 1869 (26.5%)
       Churn=No         : 5174 (73.5%)

    STEP 2: PREPROCESSING
       TotalCharges NaN : 11 imputed with median ($1397.47)
       Features after encoding : 30
       NaN in feature matrix: 0
       Train : (5634, 30)  |  Test : (1409, 30)

    OPTIMISER: SGD (Baseline)
       Epochs run       : 100
       Training time    : 86.8s
       Accuracy         : 79.06%
       Precision        : 61.93%  (Churn=Yes, pos_label=1)
       Recall           : 54.81%  (Churn=Yes, pos_label=1)
       F1-Score         : 58.16%  (Churn=Yes, pos_label=1)
       AUC-ROC          : 83.82%
       Best val_loss    : 0.5300

    OPTIMISER: Adam
       Epochs run       : 60
       Training time    : 55.2s
       Accuracy         : 79.21%
       Precision        : 61.74%
       Recall           : 56.95%
       F1-Score         : 59.25%
       AUC-ROC          : 83.78%
       Best val_loss    : 0.4788

    OPTIMISER: RMSProp
       Epochs run       : 51
       Training time    : 43.0s
       Accuracy         : 79.56%
       Precision        : 62.43%
       Recall           : 57.75%
       F1-Score         : 60.00%
       AUC-ROC          : 83.81%
       Best val_loss    : 0.4826

    WINNER SELECTION
       Best optimiser    : RMSProp
       F1 (Churn)        : 60.00%
       Recall (Churn)    : 57.75%
       AUC-ROC           : 83.81%

    All outputs saved in: otomoto_outputs/
    Application completed successfully.


### Step 6 — View output plots
All 6 PNG files appear in the otomoto_outputs/ folder.
Open them in VS Code (click the file) or in Windows File Explorer.

---

## Model Architecture

    Input Layer  (30 encoded features)
         |
    Dense(64) — L2(0.001) regularisation
    BatchNormalization
    ReLU activation
    Dropout(30%)
         |
    Dense(32) — L2(0.001)
    BatchNormalization
    ReLU activation
    Dropout(30%)
         |
    Dense(16) — L2(0.001)
    BatchNormalization
    ReLU activation
    Dropout(20%)
         |
    Output: Dense(1) — Sigmoid -> P(Churn=Yes)

Loss function : Binary cross-entropy
Callbacks     : EarlyStopping (patience=15, restore_best_weights=True)
                ReduceLROnPlateau (patience=7, factor=0.5, min_lr=1e-6)

Architecture is IDENTICAL across all three optimiser experiments.
Differences in metrics are attributable solely to the optimiser.

---

## Optimiser Configuration

    SGD      : learning_rate=0.01, momentum=0.9, nesterov=True
    Adam     : learning_rate=0.001 (defaults: beta_1=0.9, beta_2=0.999)
    RMSProp  : learning_rate=0.001, rho=0.9

---

## Why pos_label=1 Matters

All sklearn metric calls use pos_label=1 explicitly:

    precision_score(y_test, y_pred, pos_label=1)
    recall_score(y_test, y_pred,    pos_label=1)
    f1_score(y_test, y_pred,        pos_label=1)

Churn=Yes is encoded as 1 (the positive/minority class).
This ensures all reported metrics measure churn detection performance,
not retention (No-churn) performance.

This was the critical fix from Module 5 feedback, where the default
pos_label=1 for a differently-encoded dataset caused benign-class
metrics to be reported as the "priority" metric.

---

## Class Imbalance Note

class_weight={0: 1.0, 1: 2.77} was tested during development to address
the 26.5%/73.5% class imbalance. This caused model collapse across all three
optimisers (all-zero predictions). The imbalance is moderate rather than
severe; binary cross-entropy without weighting achieves F1=60% on Churn=Yes.

Future alternatives: SMOTE oversampling, focal loss.
This decision is documented in the code comments.

---

## Output Files Explained

| File                      | What it shows                                          |
|---------------------------|--------------------------------------------------------|
| 00_data_overview.png      | EDA: churn class balance + monthly charges by churn    |
| 01_training_curves.png    | Loss & accuracy curves for all 3 optimisers            |
| 02_metrics_comparison.png | Grouped bar chart: accuracy, precision, recall, F1, AUC|
| 03_roc_curves.png         | Overlaid ROC curves with AUC per optimiser             |
| 04_confusion_matrices.png | Confusion matrix for each optimiser (pos_label=1)      |
| 05_summary_table.png      | Full table with best-per-metric highlighted in green   |
| metrics.json              | All metrics saved as JSON for reference                |

---

## Troubleshooting

ModuleNotFoundError: No module named 'tensorflow'
    pip install tensorflow

ModuleNotFoundError: No module named 'seaborn'
    pip install seaborn

FileNotFoundError: teleconnect.csv not found
    Ensure teleconnect.csv is in the SAME folder as otomoto_ann_optimisation.py.
    The code reads it with pd.read_csv('teleconnect.csv') — no path prefix.

Plots not appearing as pop-up windows
    Expected. The code uses matplotlib Agg backend which saves plots as PNG
    files rather than displaying windows. Check the otomoto_outputs/ folder.

TensorFlow GPU/CPU warnings in terminal
    Harmless. Suppressed by:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  (already in the code)

---

## Key References

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
ICLR 2015. https://arxiv.org/abs/1412.6980

Tieleman, T., & Hinton, G. (2012). RMSProp: Lecture 6e, Neural Networks
for Machine Learning. COURSERA.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep
network training by reducing internal covariate shift. ICML 2015.

---

BAN6440 Business Analytics | Nexford University | May 2026
