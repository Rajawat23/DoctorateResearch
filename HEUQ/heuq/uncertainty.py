"""
uncertainty.py
--------------
BCR, uncertainty decomposition, and evaluation metrics for HEUQ.

Reference: Singh, A., Ittoo, A., Ars, P., Vandomme, E.
"Heterogeneous Ensemble framework for Uncertainty Quantification (HEUQ) in
Operations Research." Preprint submitted to AI Open, March 2026.

Implements paper Equations (2)-(6) and the three performance metrics from
Section 4.2.1.

BCR and Uncertainty Decomposition
----------------------------------
    bcr(predictions_list)
        Basic Combination Rule: simple average across M model predictions.
        Paper Equation (2).

    total_uncertainty(ensemble_proba)
        Shannon entropy of the BCR prediction.
        Paper Equation (3).

    epistemic_uncertainty(predictions_list, ensemble_proba)
        Average KL divergence between each member and the BCR.
        Equals Jensen-Shannon divergence across M distributions.
        Paper Equation (4).

    aleatoric_uncertainty(u_t, u_e)
        u_t - u_e; irreducible noise inherent in the data.
        Paper Equation (5).

Performance Metrics (Section 4.2.1)
-------------------------------------
    balanced_accuracy(y_true, y_pred)
    negative_log_likelihood(y_true, proba)
    brier_score(y_true, proba)

Bootstrap Ensemble Training
----------------------------
    train_bootstrapped_ensemble(model_factory, X_train, y_train, ...)
        Train a homogeneous baseline ensemble on stratified 90% subsets.
        Reproduces Table 5 baselines (paper Section 4.2.2).
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# BCR and uncertainty decomposition (Equations 2-6)
# ---------------------------------------------------------------------------

def bcr(predictions_list):
    """Basic Combination Rule: simple average of M model predictions.

    Paper Equation (2):
        p_BCR(y_k | x) = (1/M) * sum_{j=1}^{M} p(y_k | h_j, x)

    Parameters
    ----------
    predictions_list : list of M np.ndarray, each of shape [N, 2]
        Probability arrays from each ensemble member.

    Returns
    -------
    np.ndarray of shape [N, 2]
        Ensemble-averaged probability distribution.
    """
    return np.mean(predictions_list, axis=0)


def total_uncertainty(ensemble_proba):
    """Total uncertainty as Shannon entropy of the BCR prediction.

    Paper Equation (3):
        u_t(x) = -sum_{k} p_BCR(y_k | x) * log2( p_BCR(y_k | x) )

    Maximum entropy for binary classification is 1 bit (uniform distribution).

    Parameters
    ----------
    ensemble_proba : np.ndarray of shape [N, 2]
        BCR output from bcr().

    Returns
    -------
    np.ndarray of shape [N]
        Per-sample total uncertainty in bits.
    """
    p = np.clip(ensemble_proba, 1e-9, 1 - 1e-9)
    return -np.sum(p * np.log2(p), axis=1)


def epistemic_uncertainty(predictions_list, ensemble_proba):
    """Epistemic uncertainty as average KL divergence from BCR.

    Paper Equation (4):
        u_e(x) = (1/M) * sum_{j=1}^{M} KL[ p(. | h_j, x) || p_BCR(. | x) ]

    This quantity equals the Jensen-Shannon divergence across the M member
    distributions.  High u_e signals strong disagreement among ensemble
    members, indicating that the prediction is driven by model uncertainty
    (lack of data) rather than inherent class overlap.  In the sub-portfolio
    application (paper Section 5.4), high u_e identifies customers whose
    churn probability is uncertain due to limited training signal.

    Parameters
    ----------
    predictions_list : list of M np.ndarray, each of shape [N, 2]
    ensemble_proba : np.ndarray of shape [N, 2]
        BCR output.

    Returns
    -------
    np.ndarray of shape [N]
        Per-sample epistemic uncertainty in bits.
    """
    kl_divergences = []
    for proba in predictions_list:
        p = np.clip(proba, 1e-9, 1 - 1e-9)
        q = np.clip(ensemble_proba, 1e-9, 1 - 1e-9)
        kl = np.sum(p * np.log2(p / q), axis=1)
        kl_divergences.append(kl)
    return np.mean(kl_divergences, axis=0)


def aleatoric_uncertainty(u_t, u_e):
    """Aleatoric uncertainty: irreducible noise in the data.

    Paper Equation (5):
        u_a(x) = u_t(x) - u_e(x)

    Equivalent to the average Shannon entropy of individual member predictions
    (paper Equation 6).  Always non-negative because Jensen-Shannon divergence
    is bounded by the mixture entropy.

    High u_a indicates that even if all models agree, the label is inherently
    ambiguous (e.g., overlapping classes in feature space).

    Parameters
    ----------
    u_t : np.ndarray of shape [N]
    u_e : np.ndarray of shape [N]

    Returns
    -------
    np.ndarray of shape [N]
        Per-sample aleatoric uncertainty in bits.
    """
    return u_t - u_e


# ---------------------------------------------------------------------------
# Performance metrics (Section 4.2.1)
# ---------------------------------------------------------------------------

def balanced_accuracy(y_true, y_pred):
    """Balanced Accuracy = (TPR + TNR) / 2.

    Appropriate for imbalanced classification problems (paper uses BA throughout,
    consistent with the churn prediction literature).

    Parameters
    ----------
    y_true : array-like of shape [N]
    y_pred : array-like of shape [N]
        Hard class predictions (0 or 1).

    Returns
    -------
    float in [0, 1]
    """
    return balanced_accuracy_score(y_true, y_pred)


def negative_log_likelihood(y_true, proba):
    """Binary Negative Log-Likelihood.

    NLL = -(1/N) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    Lower NLL indicates a predictive distribution that better captures the
    variability in the true labels (paper Section 4.2.1).

    Parameters
    ----------
    y_true : np.ndarray of shape [N], values in {0, 1}
    proba : np.ndarray of shape [N, 2] or [N]
        If 2-D, column 1 is P(y=1|x).

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    if proba.ndim == 2:
        p = np.clip(proba[:, 1], 1e-9, 1 - 1e-9)
    else:
        p = np.clip(proba, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def brier_score(y_true, proba):
    """Brier Score: mean squared error between predicted probabilities and labels.

    BS = (1/N) * sum_i (p_i - y_i)^2

    Lower BS indicates better calibration.  Ranges from 0 (perfect) to 1
    (worst possible).

    Parameters
    ----------
    y_true : np.ndarray of shape [N], values in {0, 1}
    proba : np.ndarray of shape [N, 2] or [N]
        If 2-D, column 1 is P(y=1|x).

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    if proba.ndim == 2:
        p = proba[:, 1]
    else:
        p = np.asarray(proba)
    return float(np.mean((p - y_true) ** 2))


# ---------------------------------------------------------------------------
# Bootstrapped homogeneous ensemble training (paper Section 4.2.2)
# ---------------------------------------------------------------------------

def train_bootstrapped_ensemble(
    model_factory,
    X_train,
    y_train,
    n_models=10,
    subsample_rate=0.9,
    random_state=42,
):
    """Train a homogeneous bootstrapped ensemble.

    Each of the n_models models is trained on a different stratified
    subsample_rate fraction of X_train.  This follows the procedure of
    Shaker and Hullermeier (2020) as described in paper Section 4.2.2, and is
    used to create the homogeneous baseline ensembles for paper Table 5.

    Parameters
    ----------
    model_factory : callable
        A zero-argument callable that returns a fresh untrained model.
        Example: lambda: RandomForestClassifier(n_estimators=300, max_depth=9)
    X_train : np.ndarray of shape [N, D]
    y_train : np.ndarray of shape [N]
    n_models : int, optional
        Number of ensemble members. Default 10.
    subsample_rate : float, optional
        Fraction of training data per model. Default 0.9 (paper setting).
    random_state : int, optional
        Base random seed for reproducibility. Default 42.

    Returns
    -------
    list of n_models fitted model objects
    """
    models = []
    sss = StratifiedShuffleSplit(
        n_splits=n_models,
        test_size=1.0 - subsample_rate,
        random_state=random_state,
    )
    for train_idx, _ in sss.split(X_train, y_train):
        model = model_factory()
        model.fit(X_train[train_idx], y_train[train_idx])
        models.append(model)
    return models
