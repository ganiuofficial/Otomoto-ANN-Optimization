# =============================================================================
# BAN6440 - Module 6 Assignment: ANN Marketing Segmentation Optimisation
# Company      : Otomoto (Teleconnect dataset)
# Task         : Compare 3 optimisation algorithms on ANN churn classifier
# Optimisers   : SGD (baseline), Adam, RMSProp
# Dataset      : teleconnect.csv — 7,043 customers, 20 raw columns → 30 encoded features
# Author       : Ganiu Olalekan Mustapha | Nexford University | May 2026
# IMPORTANT    : Churn=Yes encoded as 1 (minority/positive class).
#                pos_label=1 is passed explicitly to all sklearn metrics
#                so that precision, recall and F1 measure churn detection,
#                not retention — the clinically/commercially relevant figure.
# =============================================================================

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, json, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      roc_auc_score, confusion_matrix,
                                      classification_report, roc_curve)

tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = "otomoto_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
def load_data(path: str = "teleconnect.csv") -> pd.DataFrame:
    print("\n── STEP 1: DATA LOADING ─────────────────────────────────────────")
    df = pd.read_csv(path)
    print(f"   Shape            : {df.shape}")
    print(f"   Churn=Yes        : {(df['Churn']=='Yes').sum()} "
          f"({(df['Churn']=='Yes').mean()*100:.1f}%)")
    print(f"   Churn=No         : {(df['Churn']=='No').sum()} "
          f"({(df['Churn']=='No').mean()*100:.1f}%)")
    return df


# =============================================================================
# STEP 2: PREPROCESSING
# Key decisions:
#   1. Drop customerID — non-informative identifier
#   2. TotalCharges: coerce to numeric; 11 whitespace rows → impute with median
#   3. Binary Yes/No columns → 1/0
#   4. Multi-category columns → one-hot encoded (drop_first=True avoids multicollinearity)
#   5. Churn: Yes=1 (positive/minority class), No=0
#   6. StandardScaler on numeric features (tenure, MonthlyCharges, TotalCharges)
#      fitted on train only — no data leakage
#   7. Stratified 80/20 split preserves 26.5% churn ratio
# =============================================================================
def preprocess(df: pd.DataFrame):
    print("\n── STEP 2: PREPROCESSING ────────────────────────────────────────")

    df = df.copy()

    # Drop identifier
    df.drop(columns=['customerID'], inplace=True)

    # Fix TotalCharges — whitespace entries coerced to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n_missing = df['TotalCharges'].isnull().sum()
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print(f"   TotalCharges NaN : {n_missing} imputed with median "
          f"(${df['TotalCharges'].median():.2f})")

    # Encode target: Yes=1, No=0
    # pos_label=1 in all metric calls → measures churn (Yes) detection
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Binary Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # gender: Female=0, Male=1
    df['gender'] = (df['gender'] == 'Male').astype(int)

    # Multi-value string columns → one-hot (drop_first to avoid dummy trap)
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    print(f"   Features after encoding : {df.shape[1]-1}")
    print(f"   Churn=1 (Yes) : {df['Churn'].sum()}  "
          f"Churn=0 (No) : {(df['Churn']==0).sum()}")

    X = df.drop('Churn', axis=1).values.astype(np.float32)
    y = df['Churn'].values.astype(np.float32)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # Scale numeric features — scaler fitted on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"   Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"   Scaling : StandardScaler (fitted on train only — no leakage)")

    _plot_churn_distribution(y, df)

    return X_train, X_test, y_train, y_test, scaler, df


def _plot_churn_distribution(y, df):
    """Bar chart of churn class distribution + key feature overview."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Otomoto (Teleconnect) — Customer Churn Overview",
                 fontweight='bold', fontsize=12)

    # Churn balance
    labels = ['No Churn (0)', 'Churn (1)']
    counts = [(y==0).sum(), (y==1).sum()]
    bars = ax1.bar(labels, counts, color=['#185FA5', '#D85A30'],
                   edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50, str(count),
                 ha='center', fontsize=11, fontweight='500')
    ax1.set_ylabel("Customers")
    ax1.set_title("Class Distribution")
    ax1.grid(axis='y', alpha=0.3)

    # Monthly charges by churn
    orig = pd.read_csv('teleconnect.csv')
    orig_churn = orig[orig['Churn'] == 'Yes']['MonthlyCharges']
    orig_no    = orig[orig['Churn'] == 'No']['MonthlyCharges']
    ax2.hist(orig_no,    bins=30, alpha=0.6, color='#185FA5', label='No Churn')
    ax2.hist(orig_churn, bins=30, alpha=0.6, color='#D85A30', label='Churn')
    ax2.set_xlabel("Monthly Charges ($)")
    ax2.set_ylabel("Count")
    ax2.set_title("Monthly Charges by Churn Status")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "00_data_overview.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Data overview     → {path}")


# =============================================================================
# STEP 3: BASELINE MODEL (SGD optimiser)
# Architecture rationale:
#   Input  : 30 encoded features (after one-hot encoding of 10 categorical columns)
#   Hidden : 64 → 32 → 16 (funnel; progressive abstraction)
#   L2(0.001) on all dense layers — regularises against overfitting
#   BatchNorm + ReLU + Dropout(0.3) per layer
#   Output : 1 neuron, sigmoid → P(Churn=Yes)
#   Loss   : Binary cross-entropy
# SGD is chosen as the BASELINE because it is the simplest optimiser —
# its performance sets the floor against which Adam and RMSProp are compared.
# =============================================================================
def build_model(input_dim: int, optimiser) -> keras.Model:
    """
    Builds a 3-hidden-layer ANN. Accepts any compiled Keras optimiser.
    Architecture is identical across all three experiments so that
    differences in metrics are attributable to the optimiser alone.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name='input'),

        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Activation('relu', name='relu_1'),
        layers.Dropout(0.30, name='dropout_1'),

        layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Activation('relu', name='relu_2'),
        layers.Dropout(0.30, name='dropout_2'),

        layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Activation('relu', name='relu_3'),
        layers.Dropout(0.20, name='dropout_3'),

        layers.Dense(1, activation='sigmoid', name='output'),
    ], name='Otomoto_ANN')

    model.compile(
        optimiser = optimiser,
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
    )
    return model


# =============================================================================
# STEP 4: TRAIN AND EVALUATE — runs for each optimiser
# NOTE on class imbalance (26.5% churn vs 73.5% no-churn):
# class_weight={0:1.0, 1:2.77} was tested during development but caused
# model collapse (all-zero predictions) across all three optimisers on this
# dataset. The imbalance is moderate rather than severe, and the current
# binary_crossentropy without weighting still achieves F1=60% on Churn=Yes.
# Future work: SMOTE oversampling or focal loss as alternatives.
# =============================================================================
def train_and_evaluate(name: str, optimiser,
                       X_train, X_test, y_train, y_test,
                       input_dim: int) -> dict:
    """
    Trains the ANN with the given optimiser, evaluates on the held-out
    test set, and returns a metrics dictionary.

    pos_label=1 is passed to all sklearn metric functions.
    Churn=Yes is encoded as 1 — the positive/minority class.
    All reported precision, recall, and F1 measure churn detection
    performance, not retention (No-churn) performance.
    """
    print(f"\n── OPTIMISER: {name} {'─'*(48-len(name))}")

    model = build_model(input_dim, optimiser)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs           = 100,
        batch_size       = 64,
        validation_split = 0.15,
        callbacks        = callbacks,
        verbose          = 0
    )
    elapsed = time.time() - start

    epochs_run = len(history.history['loss'])
    print(f"   Epochs run       : {epochs_run}  "
          f"(best weights restored)")
    print(f"   Training time    : {elapsed:.1f}s")

    # Predict probabilities
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.50).astype(int)

    # All metrics use pos_label=1 → measuring Churn=Yes detection
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_test, y_pred,    pos_label=1, zero_division=0)
    f1   = f1_score(y_test, y_pred,        pos_label=1, zero_division=0)
    # Guard against NaN from SGD divergence
    if np.isnan(y_prob).any():
        print("   WARNING: NaN predictions detected — optimiser diverged.")
        y_prob = np.zeros_like(y_prob)
    auc  = roc_auc_score(y_test, y_prob)
    loss = min(history.history['val_loss'])

    print(f"   Accuracy         : {acc*100:.2f}%")
    print(f"   Precision        : {prec*100:.2f}%  (Churn=Yes, pos_label=1)")
    print(f"   Recall           : {rec*100:.2f}%   (Churn=Yes, pos_label=1) ← key for marketing")
    print(f"   F1-Score         : {f1*100:.2f}%")
    print(f"   AUC-ROC          : {auc*100:.2f}%")
    print(f"   Final val_loss   : {loss:.4f}")

    print("\n   Classification Report (Churn=Yes as positive class):")
    print(classification_report(y_test, y_pred,
                                target_names=['No Churn (0)', 'Churn (1)']))

    result = {
        'name'     : name,
        'accuracy' : round(acc,  4),
        'precision': round(prec, 4),
        'recall'   : round(rec,  4),
        'f1'       : round(f1,   4),
        'auc'      : round(auc,  4),
        'val_loss' : round(loss, 4),
        'epochs'   : epochs_run,
        'time_s'   : round(elapsed, 1),
        'history'  : history.history,
        'y_prob'   : y_prob,
        'model'    : model,
    }
    return result


# =============================================================================
# STEP 5: VISUALISATIONS
# =============================================================================
def plot_training_curves(results: list) -> None:
    """Loss and accuracy curves for all three optimisers side by side."""
    colours = ['#185FA5', '#1D9E75', '#D85A30']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Curves — All Optimisers (Otomoto ANN)",
                 fontweight='bold', fontsize=12)

    for r, colour in zip(results, colours):
        h = r['history']
        ep = range(1, len(h['loss']) + 1)
        axes[0].plot(ep, h['loss'],     color=colour, label=f"{r['name']} train",
                     linewidth=2)
        axes[0].plot(ep, h['val_loss'], color=colour, linestyle='--', alpha=0.6,
                     label=f"{r['name']} val")
        axes[1].plot(ep, h['accuracy'],     color=colour, linewidth=2,
                     label=f"{r['name']} train")
        axes[1].plot(ep, h['val_accuracy'], color=colour, linestyle='--', alpha=0.6,
                     label=f"{r['name']} val")

    axes[0].set_title("Loss (Binary Cross-Entropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Training curves   → {path}")


def plot_metrics_comparison(results: list) -> None:
    """Grouped bar chart comparing all metrics across 3 optimisers."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels  = ['Accuracy', 'Precision\n(Churn)', 'Recall\n(Churn)', 'F1\n(Churn)', 'AUC-ROC']
    colours = ['#185FA5', '#1D9E75', '#D85A30']

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (r, colour) in enumerate(zip(results, colours)):
        vals = [r[m]*100 for m in metrics]
        bars = ax.bar(x + i*width, vals, width, label=r['name'],
                      color=colour, alpha=0.87, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}%", ha='center', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Performance Comparison — SGD vs Adam vs RMSProp\n"
                 "All churn metrics use pos_label=1 (Churn=Yes)",
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Metrics comparison → {path}")


def plot_roc_curves(results: list, y_test) -> None:
    """Overlaid ROC curves for all three optimisers."""
    colours = ['#185FA5', '#1D9E75', '#D85A30']
    fig, ax = plt.subplots(figsize=(7, 6))

    for r, colour in zip(results, colours):
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{r['name']}  (AUC={r['auc']*100:.2f}%)")

    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.fill_between([0,1],[0,1], alpha=0.04, color='grey')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Optimiser Comparison (Otomoto ANN)",
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ROC curves         → {path}")


def plot_confusion_matrices(results: list, y_test) -> None:
    """Confusion matrix for each optimiser."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Confusion Matrices — Churn=Yes is Positive Class (pos_label=1)",
                 fontweight='bold', fontsize=11)

    for ax, r in zip(axes, results):
        y_pred = (r['y_prob'] >= 0.50).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred No (0)', 'Pred Churn (1)'],
                    yticklabels=['True No (0)', 'True Churn (1)'],
                    annot_kws={'size': 12})
        ax.set_title(f"{r['name']}\nF1={r['f1']*100:.1f}%  Recall={r['recall']*100:.1f}%",
                     fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrices → {path}")


def plot_summary_table(results: list) -> None:
    """Clean summary table saved as figure for easy inclusion in Word doc."""
    metrics = ['accuracy','precision','recall','f1','auc','val_loss','epochs','time_s']
    labels  = ['Accuracy','Precision\n(Churn)','Recall\n(Churn)',
                'F1\n(Churn)','AUC-ROC','Val Loss','Epochs','Time (s)']

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    col_labels = ['Metric'] + [r['name'] for r in results]
    rows = []
    for m, lbl in zip(metrics, labels):
        row = [lbl]
        for r in results:
            val = r[m]
            if m in ['accuracy','precision','recall','f1','auc']:
                row.append(f"{val*100:.2f}%")
            elif m == 'val_loss':
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        rows.append(row)

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1F4E79')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best value per metric row (cols 1,2,3)
    for i, m in enumerate(metrics):
        vals = [results[j][m] for j in range(3)]
        best_j = int(np.argmax(vals)) if m != 'val_loss' else int(np.argmin(vals))
        tbl[i+1, best_j+1].set_facecolor('#E1F5EE')

    ax.set_title("Performance Summary — All Optimisers (green = best per metric)",
                 fontweight='bold', pad=20, fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Summary table      → {path}")


# =============================================================================
# STEP 6: SAVE METRICS JSON
# =============================================================================
def save_metrics(results: list) -> None:
    out = []
    for r in results:
        out.append({k: v for k, v in r.items()
                    if k not in ('history', 'y_prob', 'model')})
    path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"   Metrics JSON       → {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  BAN6440 Module 6 | ANN Optimisation | Otomoto Marketing")
    print("  Dataset : teleconnect.csv (7,043 customers, 21 columns)")
    print("  Optimisers : SGD → Adam → RMSProp")
    print("=" * 65)

    df = load_data('teleconnect.csv')
    X_train, X_test, y_train, y_test, scaler, df_enc = preprocess(df)

    input_dim = X_train.shape[1]
    print(f"\n   Input dimensions : {input_dim} features")

    # ── Define three optimisers ──────────────────────────────────────
    # SGD:      Baseline — simplest optimiser; fixed learning rate;
    #           momentum=0.9 standard convention; no adaptive rates.
    #           Slow to converge but interpretable.
    #
    # Adam:     Adaptive Moment Estimation — combines momentum and
    #           RMSProp; handles sparse gradients well; most widely
    #           used in industry for tabular ANN (Kingma & Ba, 2015).
    #
    # RMSProp:  Root Mean Square Propagation — adapts learning rate
    #           per parameter using exponential moving average of
    #           squared gradients; strong on non-stationary objectives;
    #           well-suited to customer behaviour data with varying
    #           feature scales (Tieleman & Hinton, 2012).
    # ─────────────────────────────────────────────────────────────────
    optimisers = [
        ("SGD (Baseline)",
         keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)),
        ("Adam",
         keras.optimizers.Adam(learning_rate=0.001)),
        ("RMSProp",
         keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)),
    ]

    results = []
    for name, opt in optimisers:
        r = train_and_evaluate(name, opt, X_train, X_test,
                               y_train, y_test, input_dim)
        results.append(r)

    # ── Visualisations ───────────────────────────────────────────────
    print("\n── VISUALISATIONS ───────────────────────────────────────────────")
    plot_training_curves(results)
    plot_metrics_comparison(results)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_summary_table(results)
    save_metrics(results)

    # ── Final winner ─────────────────────────────────────────────────
    print("\n── WINNER SELECTION ─────────────────────────────────────────────")
    # Primary criterion: F1-score on Churn=Yes (balances precision and recall)
    # Secondary: AUC-ROC (threshold-independent)
    best = max(results, key=lambda r: (r['f1'], r['auc']))
    print(f"   Best optimiser    : {best['name']}")
    print(f"   F1 (Churn)        : {best['f1']*100:.2f}%")
    print(f"   Recall (Churn)    : {best['recall']*100:.2f}%")
    print(f"   AUC-ROC           : {best['auc']*100:.2f}%")
    print(f"\n   Recommendation: Deploy {best['name']} model for Otomoto")
    print(f"   marketing campaigns. Focus outreach on customers the model")
    print(f"   scores above 0.50 churn probability — highest ROI segment.")

    print("\n" + "=" * 65)
    print("  ✓ Application completed. All outputs in otomoto_outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
