"""
heuq
----
HEUQ: Heterogeneous Ensemble framework for Uncertainty Quantification.

Reference: Singh, A., Ittoo, A., Ars, P., Vandomme, E.
"Heterogeneous Ensemble framework for Uncertainty Quantification (HEUQ) in
Operations Research." Preprint submitted to AI Open, March 2026.

Public API
----------
Model training and prediction:
    train_model            -- Train a single HEUQ base classifier.
    predict_proba          -- Return [N, 2] probability array.
    save_predictions       -- Persist [N, 2] array to CSV.
    load_predictions       -- Load [N, 2] array from CSV.

BCR and uncertainty decomposition (paper Equations 2-6):
    bcr                    -- Basic Combination Rule (simple average).
    total_uncertainty      -- Shannon entropy of BCR prediction.
    epistemic_uncertainty  -- Average KL divergence from BCR.
    aleatoric_uncertainty  -- u_t - u_e.

Performance metrics (paper Section 4.2.1):
    balanced_accuracy
    negative_log_likelihood
    brier_score

Bootstrapped ensemble training (paper Section 4.2.2):
    train_bootstrapped_ensemble

Dimensionality reduction (paper Section 4.1.3):
    PCATransform
    GaussianRandomProjection

Ablation study (paper Table 6):
    ablation_study

"""

from .models import train_model, predict_proba, save_predictions, load_predictions
from .uncertainty import (
    bcr,
    total_uncertainty,
    epistemic_uncertainty,
    aleatoric_uncertainty,
    balanced_accuracy,
    negative_log_likelihood,
    brier_score,
    train_bootstrapped_ensemble,
)
from .transforms import PCATransform, GaussianRandomProjection
from .ablation import ablation_study

__all__ = [
    "train_model",
    "predict_proba",
    "save_predictions",
    "load_predictions",
    "bcr",
    "total_uncertainty",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "balanced_accuracy",
    "negative_log_likelihood",
    "brier_score",
    "train_bootstrapped_ensemble",
    "PCATransform",
    "GaussianRandomProjection",
    "ablation_study",
]
