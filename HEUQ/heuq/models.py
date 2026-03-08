"""
models.py
---------
HEUQ base classifiers: uniform training and prediction interface for all six
heterogeneous ensemble members used in the paper.

Reference: Singh, A., Ittoo, A., Ars, P., Vandomme, E.
"Heterogeneous Ensemble framework for Uncertainty Quantification (HEUQ) in
Operations Research." Preprint submitted to AI Open, March 2026.

Six classifiers (paper Section 3.3 and Section 4.2.2):
    LR   -- Logistic Regression
    RF   -- Random Forest
    BG   -- Bagging (Decision Tree base)
    XGB  -- XGBoost
    CB   -- CatBoost
    DNN  -- Deep Neural Network (PyTorch, 2-class softmax output)

All models accept arbitrary np.ndarray inputs.  No hardcoded data paths.

Public API
----------
    train_model(model_name, X_train, y_train, ...)
    predict_proba(model, X_test)
    save_predictions(proba_array, filepath)
    load_predictions(filepath)
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ---------------------------------------------------------------------------
# DNN definition (paper Section 4.2.2)
# ---------------------------------------------------------------------------

class _DNNClassifier(nn.Module):
    """Two-hidden-layer neural network for binary classification.

    Architecture (paper Section 4.2.2):
        Linear(n_features, 2048) -> ReLU -> Dropout(0.5)
        Linear(2048, 256)        -> ReLU -> Dropout(0.5)
        Linear(256, 2)           -> Softmax

    Parameters
    ----------
    n_features : int
        Input dimension.
    dropout : float, optional
        Dropout probability. Default 0.5.
    """

    def __init__(self, n_features, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


def _train_dnn(X_train, y_train, class_weight=None, hyperparams=None):
    """Train the DNN and return a fitted (_DNNClassifier, device) tuple.

    Paper defaults: epochs=30, lr=0.00025, weight_decay=0.0001, dropout=0.5.
    """
    hp = {
        "epochs": 30,
        "lr": 0.00025,
        "weight_decay": 0.0001,
        "dropout": 0.5,
        "batch_size": 256,
    }
    if hyperparams:
        hp.update(hyperparams)

    device = torch.device("cpu")
    n_features = X_train.shape[1]
    model = _DNNClassifier(n_features, dropout=hp["dropout"]).to(device)

    # Weighted binary cross-entropy for imbalanced data
    if class_weight is not None:
        if isinstance(class_weight, dict):
            w = torch.tensor(
                [class_weight.get(0, 1.0), class_weight.get(1, 1.0)],
                dtype=torch.float32,
            ).to(device)
        else:
            w = torch.tensor(class_weight, dtype=torch.float32).to(device)
    else:
        w = None

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train if isinstance(y_train, np.ndarray) else y_train.values,
                       dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True)

    model.train()
    for epoch in range(hp["epochs"]):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    return model, device


# ---------------------------------------------------------------------------
# Sklearn model constructors (paper Section 4.2.2)
# ---------------------------------------------------------------------------

_SKLEARN_DEFAULTS = {
    "LR": {
        "max_iter": 1000,
        "penalty": "l2",
        "solver": "lbfgs",
    },
    "RF": {
        "n_estimators": 300,
        "max_depth": 9,
        "n_jobs": -1,
    },
    "BG": {
        "max_samples": 0.3,
        "n_estimators": 700,
        "n_jobs": -1,
    },
    "XGB": {
        "n_estimators": 300,
        "learning_rate": 1.0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    },
    "CB": {
        "depth": 5,
        "learning_rate": 0.1,
        "iterations": 1000,
        "random_state": 42,
        "logging_level": "Silent",
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
    },
}


def _build_sklearn_model(model_name, class_weight, hyperparams):
    hp = dict(_SKLEARN_DEFAULTS.get(model_name, {}))
    if hyperparams:
        hp.update(hyperparams)

    if model_name == "LR":
        return LogisticRegression(class_weight=class_weight or "balanced", **hp)

    if model_name == "RF":
        return RandomForestClassifier(class_weight=class_weight or "balanced", **hp)

    if model_name == "BG":
        base = DecisionTreeClassifier(max_depth=6)
        return BaggingClassifier(base, **hp)

    if model_name == "XGB":
        # XGBoost uses scale_pos_weight instead of class_weight
        if class_weight and isinstance(class_weight, dict):
            n_neg = class_weight.get(0, 1)
            n_pos = class_weight.get(1, 1)
            hp["scale_pos_weight"] = n_neg / max(n_pos, 1)
        return XGBClassifier(**hp)

    if model_name == "CB":
        cw = [1, 12]   # paper default for churn minority class weight
        if class_weight and isinstance(class_weight, (list, tuple)):
            cw = class_weight
        return CatBoostClassifier(class_weights=cw, **hp)

    raise ValueError("Unknown model_name: %s. Choose from LR, RF, BG, XGB, CB, DNN." % model_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(model_name, X_train, y_train, class_weight=None, hyperparams=None):
    """Train a single HEUQ base classifier and return the fitted object.

    Parameters
    ----------
    model_name : str
        One of "LR", "RF", "BG", "XGB", "CB", "DNN".
    X_train : np.ndarray of shape [N, D]
        Training features (already preprocessed: scaled numerics, encoded categoricals).
    y_train : np.ndarray of shape [N]
        Binary labels (0 or 1).
    class_weight : dict, list, or None, optional
        Class weights for imbalanced data.
        For LR, RF, BG: dict {0: w0, 1: w1} or 'balanced'.
        For XGB: dict {0: n_neg, 1: n_pos} (converted to scale_pos_weight).
        For CB: list [w0, w1] (paper default [1, 12]).
        For DNN: dict {0: w0, 1: w1} or list [w0, w1].
    hyperparams : dict or None, optional
        Override default hyperparameters.

    Returns
    -------
    Fitted model object (sklearn estimator or (_DNNClassifier, device) tuple).
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    if model_name == "DNN":
        return _train_dnn(X_train, y_train, class_weight=class_weight, hyperparams=hyperparams)

    model = _build_sklearn_model(model_name, class_weight, hyperparams)
    model.fit(X_train, y_train)
    return model


def predict_proba(model, X_test):
    """Return probability array of shape [N, 2] where column 1 is P(y=1|x).

    Handles both sklearn classifiers and the DNN (returned as a tuple by
    train_model when model_name="DNN").

    Parameters
    ----------
    model : sklearn estimator or (_DNNClassifier, device) tuple
    X_test : np.ndarray of shape [N, D]

    Returns
    -------
    np.ndarray of shape [N, 2]
    """
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    if isinstance(model, tuple):
        # DNN case: (model, device)
        net, device = model
        net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            proba = net(X_t).cpu().numpy()
        return proba

    # Sklearn case
    return model.predict_proba(X_test)


def save_predictions(proba_array, filepath):
    """Save [N, 2] probability array to a CSV file.

    Parameters
    ----------
    proba_array : np.ndarray of shape [N, 2]
    filepath : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df = pd.DataFrame(proba_array, columns=["prob_class_0", "prob_class_1"])
    df.to_csv(filepath, index=False)


def load_predictions(filepath):
    """Load [N, 2] probability array from a CSV file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    np.ndarray of shape [N, 2]
    """
    df = pd.read_csv(filepath)
    return df.values
