"""
run_heuq_bank.py
----------------
Self-contained HEUQ demo using the publicly available UCI Bank Marketing dataset.

This demo reproduces the HEUQ methodology (paper Sections 4-5) on an open
dataset so that the full pipeline can be verified without access to the private
Ethias dataset used in the paper.

Dataset
-------
UCI Bank Marketing Dataset (Moro et al., 2014):
    http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    Binary target: y = 1 if the customer subscribed to a term deposit.
    Features: 16 attributes (numerical and categorical).
    Size: ~41,188 samples with 11.3% positive rate.

This dataset is structurally similar to the churn dataset used in the paper:
    - Imbalanced binary classification.
    - Mix of numerical and categorical features.
    - Similar sample size.

Pipeline
--------
    1. Load and preprocess data (one-hot encoding + MinMax scaling).
    2. Apply PCA (95% variance threshold) -> ~10 components for Bank dataset.
    3. Train 6 HEUQ base classifiers: LR, RF, BG, XGB, CB, DNN.
    4. Compute BCR predictions and uncertainty decomposition.
    5. Run leave-one-out ablation study (paper Table 6 methodology).
    6. Print summary metrics table.

Usage
-----
    cd DoctorateResearch/HEUQ
    pip install -r requirements.txt
    python demo/run_heuq_bank.py

    # Skip DNN training (faster demo):
    python demo/run_heuq_bank.py --skip_dnn

Output
------
    ./results/heuq_bank_predictions.csv
    ./results/heuq_bank_uncertainty.csv
    ./results/heuq_bank_ablation.csv
    Stdout: metrics table (BA, NLL, BS, u_t, u_e, u_a)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Make heuq importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import heuq
from heuq import (
    train_model, predict_proba, save_predictions,
    bcr, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty,
    balanced_accuracy, negative_log_likelihood, brier_score,
    PCATransform,
    ablation_study,
)


# ---------------------------------------------------------------------------
# Data loading (UCI Bank Marketing, downloaded on demand)
# ---------------------------------------------------------------------------

def _load_bank_dataset(data_dir="./data"):
    """Load or download UCI Bank Marketing dataset.

    Tries data_dir/bank-additional-full.csv first.  If not found, downloads
    from UCI ML repository.

    Returns
    -------
    X_train, X_test : np.ndarray of shape [N, D]
    y_train, y_test : np.ndarray of shape [N]
    """
    csv_path = os.path.join(data_dir, "bank-additional-full.csv")
    if not os.path.exists(csv_path):
        import urllib.request
        import zipfile
        os.makedirs(data_dir, exist_ok=True)
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00222/bank-additional.zip"
        )
        zip_path = os.path.join(data_dir, "bank-additional.zip")
        print("Downloading UCI Bank Marketing dataset from %s ..." % url)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        # the full dataset is inside the extracted subdirectory
        extracted_csv = os.path.join(
            data_dir, "bank-additional", "bank-additional-full.csv"
        )
        if os.path.exists(extracted_csv):
            import shutil
            shutil.copy(extracted_csv, csv_path)

    df = pd.read_csv(csv_path, sep=";")
    df = df.rename(columns={"y": "target"})
    df["target"] = (df["target"] == "yes").astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols)

    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Dataset: Bank Marketing  |  train=%d  test=%d  pos_rate_train=%.3f" % (
        len(y_train), len(y_test), y_train.mean()
    ))
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HEUQ Bank Marketing demo")
    parser.add_argument("--skip_dnn", action="store_true",
                        help="Skip DNN training for a faster demo run.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = _load_bank_dataset(args.data_dir)

    # 2. PCA dimensionality reduction (paper Section 4.1.3)
    pca = PCATransform(variance_threshold=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("PCA: %d components (95%% variance)" % pca.n_components_)

    # 3. Train base classifiers
    model_names = ["LR", "RF", "BG", "XGB", "CB"]
    if not args.skip_dnn:
        model_names.append("DNN")
    else:
        print("Skipping DNN.")

    predictions = {}
    for name in model_names:
        print("Training %s ..." % name)
        model = train_model(name, X_train_pca, y_train)
        proba = predict_proba(model, X_test_pca)
        predictions[name] = proba

        csv_path = os.path.join(args.results_dir, "heuq_bank_%s.csv" % name)
        save_predictions(proba, csv_path)

    # 4. BCR and uncertainty decomposition
    probas_list = list(predictions.values())
    ensemble_proba = bcr(probas_list)

    u_t = total_uncertainty(ensemble_proba)
    u_e = epistemic_uncertainty(probas_list, ensemble_proba)
    u_a = aleatoric_uncertainty(u_t, u_e)

    y_pred = (ensemble_proba[:, 1] >= 0.5).astype(int)

    # Save uncertainty arrays
    unc_df = pd.DataFrame({
        "u_t": u_t,
        "u_e": u_e,
        "u_a": u_a,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    unc_df.to_csv(
        os.path.join(args.results_dir, "heuq_bank_uncertainty.csv"), index=False
    )

    # 5. Print main metrics
    ba  = balanced_accuracy(y_test, y_pred)
    nll = negative_log_likelihood(y_test, ensemble_proba)
    bs  = brier_score(y_test, ensemble_proba)

    print("\nHEUQ (%d models) on UCI Bank Marketing" % len(model_names))
    print("  BA  = %.4f" % ba)
    print("  NLL = %.4f" % nll)
    print("  BS  = %.4f" % bs)
    print("  u_t = %.4f (mean total uncertainty)" % u_t.mean())
    print("  u_e = %.4f (mean epistemic uncertainty)" % u_e.mean())
    print("  u_a = %.4f (mean aleatoric uncertainty)" % u_a.mean())

    # 6. Ablation study
    print("\nRunning ablation study ...")
    ablation_results = ablation_study(predictions, y_test)

    ablation_rows = []
    for variant, metrics in ablation_results.items():
        ablation_rows.append({
            "variant": variant,
            "members": str(metrics["ensemble_members"]),
            "ba":  "%.4f" % metrics["ba"],
            "nll": "%.4f" % metrics["nll"],
            "bs":  "%.4f" % metrics["bs"],
            "u_t": "%.4f" % metrics["u_t"],
            "u_e": "%.4f" % metrics["u_e"],
            "u_a": "%.4f" % metrics["u_a"],
        })
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(
        os.path.join(args.results_dir, "heuq_bank_ablation.csv"), index=False
    )
    print(abl_df[["variant", "ba", "nll", "bs", "u_e"]].to_string(index=False))

    print("\nResults saved to %s/" % args.results_dir)


if __name__ == "__main__":
    main()
