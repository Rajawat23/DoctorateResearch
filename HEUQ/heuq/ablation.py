"""
ablation.py
-----------
Ablation study for HEUQ: remove one model from the 6-model ensemble and
measure the change in all uncertainty and performance metrics.

Reference: Singh, A., Ittoo, A., Ars, P., Vandomme, E.
"Heterogeneous Ensemble framework for Uncertainty Quantification (HEUQ) in
Operations Research." Preprint submitted to AI Open, March 2026.

Reproduces Table 6 of the paper (Section 5.3).

Key findings from the paper:
    - Ensembles including DNN show lower aleatoric uncertainty; the DNN captures
      complex nonlinear relationships that tree-based models miss.
    - The tree-only ensemble achieves the lowest epistemic uncertainty but the
      worst calibration (NLL and BS).
    - All HEUQ variants achieve comparable BA but differ in uncertainty estimates.
    - The full 6-model HEUQ achieves better NLL and BS than any ablated variant.
"""

import numpy as np

from .uncertainty import (
    bcr,
    total_uncertainty,
    epistemic_uncertainty,
    aleatoric_uncertainty,
    balanced_accuracy,
    negative_log_likelihood,
    brier_score,
)


def ablation_study(predictions_dict, y_true, y_pred_threshold=0.5):
    """Ablation study: leave-one-out analysis of the HEUQ ensemble.

    For each model in predictions_dict, removes it from the ensemble and
    computes all metrics (NLL, BS, BA, u_t, u_e, u_a) on the remaining
    5-model sub-ensemble.  Also reports the full 6-model metrics as baseline.

    Reproduces Table 6 of the paper.

    Key findings from the paper:
        - DNN removal increases u_a most (DNN captures non-linear aleatoric signal).
        - Tree-only ensemble has the lowest u_e but the worst NLL and BS.
        - All ablated variants have comparable BA to the full ensemble.
        - Full HEUQ consistently achieves the best NLL and BS.

    Parameters
    ----------
    predictions_dict : dict
        Mapping from model name (str) to np.ndarray of shape [N, 2].
        Expected keys: "LR", "RF", "BG", "XGB", "CB", "DNN".
    y_true : np.ndarray of shape [N]
        Ground-truth binary labels.
    y_pred_threshold : float, optional
        Probability threshold for hard predictions. Default 0.5.

    Returns
    -------
    dict
        Keys are model names (plus "full_HEUQ").  Each value is a dict with:
            ensemble_members : list of model names in the sub-ensemble
            ba               : Balanced Accuracy
            nll              : Negative Log-Likelihood
            bs               : Brier Score
            u_t              : mean total uncertainty
            u_e              : mean epistemic uncertainty
            u_a              : mean aleatoric uncertainty
    """
    model_names = list(predictions_dict.keys())
    all_probas = list(predictions_dict.values())

    def _compute_metrics(probas, member_names):
        ensemble_proba = bcr(probas)
        u_t = total_uncertainty(ensemble_proba)
        u_e = epistemic_uncertainty(probas, ensemble_proba)
        u_a = aleatoric_uncertainty(u_t, u_e)
        y_pred = (ensemble_proba[:, 1] >= y_pred_threshold).astype(int)
        return {
            "ensemble_members": member_names,
            "ba":  float(balanced_accuracy(y_true, y_pred)),
            "nll": float(negative_log_likelihood(y_true, ensemble_proba)),
            "bs":  float(brier_score(y_true, ensemble_proba)),
            "u_t": float(np.mean(u_t)),
            "u_e": float(np.mean(u_e)),
            "u_a": float(np.mean(u_a)),
        }

    results = {}

    # Full ensemble baseline
    results["full_HEUQ"] = _compute_metrics(all_probas, model_names)

    # Leave-one-out ablations
    for ablated in model_names:
        remaining_names = [n for n in model_names if n != ablated]
        remaining_probas = [predictions_dict[n] for n in remaining_names]
        label = "ablate_%s" % ablated
        results[label] = _compute_metrics(remaining_probas, remaining_names)

    return results
