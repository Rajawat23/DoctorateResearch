# DoctorateResearch

This repository contains the public research code for fulfillment of doctorate on uncertainty quantification in machine learning and reinforcement learning.

All code runs on publicly available datasets and benchmarks only.

---

## Contents

| Package | Folder | Description |
|---------|--------|-------------|
| HEUQ | `HEUQ/` | Heterogeneous Ensemble framework for Uncertainty Quantification |
| MASURE | `MEASURE/` | Masksembles for Stable and Uncertainty-aware RL Environments |
| Masksembles | `MEASURE/masksembles-main/` | Vendored upstream Masksembles library (MIT) |

---

## Papers

**HEUQ**
Singh, A., Ittoo, A., Ars, P., Vandomme, E.
"Heterogeneous Ensemble framework for Uncertainty Quantification (HEUQ) in
Operations Research."
Preprint submitted to AI Open, March 2026.

**MASURE**
Singh, A., Ittoo, A., Vandomme, E., Ars, P.
"Uncertainty-Aware Reinforcement Learning Agents for Noisy Environments."
IEEE/CVF, 2026.

---

## HEUQ

### What it does

HEUQ is a model-agnostic framework for uncertainty quantification using a heterogeneous
ensemble of six classifiers (LR, RF, BG, XGB, CatBoost, DNN). Predictions are combined
with a simple average (Basic Combination Rule). Total uncertainty is the Shannon entropy
of the ensemble prediction. Epistemic uncertainty is the average KL divergence between
each member's prediction and the ensemble prediction (Jensen-Shannon divergence).
Aleatoric uncertainty is the residual.

The framework is evaluated on churn prediction tasks (paper Tables 3, 5, 6) and includes
a sub-portfolio selection application (paper Section 5.4).

### Installation

```bash
cd HEUQ
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r HEUQ/requirements.txt
```

Requirements: `torch>=1.12.0`, `scikit-learn>=1.0.0`, `numpy`, `pandas`,
`xgboost>=1.6.0`, `catboost>=1.0.0`, `scipy`.

### Quick start

```python
import numpy as np
from heuq import (
    train_model, predict_proba,
    bcr, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty,
    balanced_accuracy, negative_log_likelihood, brier_score,
    PCATransform, ablation_study,
)

# Train six base classifiers
model_names = ["LR", "RF", "BG", "XGB", "CB", "DNN"]
predictions = {}
for name in model_names:
    model = train_model(name, X_train, y_train)
    predictions[name] = predict_proba(model, X_test)

# Ensemble prediction and uncertainty decomposition
probas = list(predictions.values())
ensemble_proba = bcr(probas)

u_t = total_uncertainty(ensemble_proba)     # paper Eq. (3)
u_e = epistemic_uncertainty(probas, ensemble_proba)  # paper Eq. (4)
u_a = aleatoric_uncertainty(u_t, u_e)       # paper Eq. (5)

# Performance metrics
y_pred = (ensemble_proba[:, 1] >= 0.5).astype(int)
print("BA  =", balanced_accuracy(y_test, y_pred))
print("NLL =", negative_log_likelihood(y_test, ensemble_proba))
print("BS  =", brier_score(y_test, ensemble_proba))
```

### Demo

The demo runs on the UCI Bank Marketing dataset (downloaded automatically on first run):

```bash
cd HEUQ
python demo/run_heuq_bank.py
```

To skip DNN training for a faster run:

```bash
python demo/run_heuq_bank.py --skip_dnn
```
### Public API

| Symbol | Description | Paper reference |
|--------|-------------|-----------------|
| `train_model(name, X, y)` | Train one of LR, RF, BG, XGB, CB, DNN | Sec. 3.3, 4.2.2 |
| `predict_proba(model, X)` | Returns `[N, 2]` probability array | Sec. 3.3 |
| `bcr(predictions_list)` | Basic Combination Rule (simple average) | Eq. (2) |
| `total_uncertainty(proba)` | Shannon entropy of BCR output | Eq. (3) |
| `epistemic_uncertainty(preds, proba)` | Average KL divergence from BCR | Eq. (4) |
| `aleatoric_uncertainty(u_t, u_e)` | u_t minus u_e | Eq. (5) |
| `balanced_accuracy(y_true, y_pred)` | (TPR + TNR) / 2 | Sec. 4.2.1 |
| `negative_log_likelihood(y_true, proba)` | Binary NLL | Sec. 4.2.1 |
| `brier_score(y_true, proba)` | Mean squared probability error | Sec. 4.2.1 |
| `train_bootstrapped_ensemble(factory, X, y)` | Homogeneous ensemble baseline | Sec. 4.2.2 |
| `PCATransform(variance_threshold)` | PCA dimensionality reduction | Sec. 4.1.3 |
| `GaussianRandomProjection(n_components, epsilon)` | Johnson-Lindenstrauss projection | Sec. 4.1.3 |
| `ablation_study(predictions_dict, y_true)` | Leave-one-out ablation | Sec. 5.3, Table 6 |
| `subportfolio_by_uncertainty(u, y_true, y_pred, q)` | Sub-portfolio filtering by uncertainty | Sec. 5.4, Table 8 |
| `precision_by_uncertainty_bins(u_e, u_a, ...)` | Precision across u_e x u_a bins | Sec. 5.4, Fig. 5, Table 7 |

### Key hyperparameters (paper Section 4.2.2)

| Model | Key hyperparameters |
|-------|---------------------|
| LR | `max_iter=1000`, `penalty=l2`, `solver=lbfgs` |
| RF | `n_estimators=300`, `max_depth=9` |
| BG | `base=DecisionTree(max_depth=6)`, `max_samples=0.3`, `n_estimators=700` |
| XGB | `n_estimators=300`, `learning_rate=1.0` |
| CB | `depth=5`, `learning_rate=0.1`, `class_weights=[1, 12]` |
| DNN | `hidden=[2048, 256]`, `dropout=0.5`, `epochs=30`, `lr=0.00025`, Adam |

---

## MASURE

### What it does

MASURE (Masksembles for Stable and Uncertainty-aware Reinforcement Learning
Environments) integrates Masksembles-based epistemic uncertainty into Q-learning.
The core contribution is an uncertainty-conscious update rule that scales the temporal
difference step inversely with epistemic variance:

```
Q_{t+1}(s, a) = Q_t(s, a) + (1 / (1 + sigma^2_t(s, a))) * alpha * delta_t
```

This dampens unstable updates during consecutive noisy states, preventing catastrophic
forgetting. MASURE achieves this at significantly lower computational cost than deep
ensembles (6.02K parameters vs 24-26K for SUNRISE/IV-DQN, paper Table I).

### Installation

Install the vendored Masksembles dependency first:

```bash
pip install -e MEASURE/masksembles-main/
```

Then install MASURE dependencies:

```bash
pip install torch>=1.12.0 gym==0.21.0 numpy scipy
```

### Training an agent

```bash
cd MEASURE

# Train MASURE on LunarLander-v2 with paper hyperparameters
python -m masure.train --model MasksembleDQN --env LunarLander-v2 --num_episodes 400

# Train IV-DQN baseline
python -m masure.train --model IVDQN --env LunarLander-v2 --num_episodes 400

# Train standard DQN baseline
python -m masure.train --model DQN --env CartPole-v1 --num_episodes 400
```

Results are written to `./results/{model}_{env}_{env_seed}_{net_seed}.csv`.

### Demo

```bash
cd MEASURE
python demo/run_masure_lunarlander.py
```

Expected result: MASURE achieves higher and more stable cumulative returns than IVDQN
under burst-style noise on LunarLander-v2 (paper Figure 1, Figure 2).

### Agents

| CLI name | Class | Description | Paper reference |
|----------|-------|-------------|-----------------|
| `MasksembleDQN` | `MASUREAgent` | Core contribution: uncertainty-conscious Q-update | Eq. (4)-(8) |
| `DQN` | `DQNAgent` | Standard DQN baseline | Table II |
| `BootstrapDQN` | `BootstrapDQNAgent` | Bootstrapped DQN with Randomized Prior Functions | Table II |
| `SunriseDQN` | `SunriseDQNAgent` | SUNRISE (Lee et al., ICML 2021) | Table II |
| `IVDQN` | `IVDQNAgent` | IV-RL (Mai et al., ICLR 2022) | Table II |

### Environments

| Environment | Description |
|-------------|-------------|
| `LunarLander-v2` | Primary benchmark; dense rewards |
| `CartPole-v1` | Balance task; MASURE outperforms all baselines |
| `MountainCar-v0` | Sparse rewards; MASURE and IVDQN both struggle |

### Noise schedule (paper Section III-D, Table II)

The `StepBurstNoiseObservation` wrapper injects rare but severe Gaussian noise bursts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_rate` | 0.01 | Per-step probability of triggering a burst |
| `scale` | 1.0 | Gaussian noise amplitude N(0, scale) |
| `burst_length` | 500 | Duration of each burst in steps |
| `burst_cooldown` | 20000 | Minimum steps between bursts |
| `start_episode` | 100 | Warm-up episodes before noise can trigger |

### Hyperparameters (paper Table II)

| Agent | eff_batch | lr | gamma | tau | Additional |
|-------|-----------|-----|-------|-----|------------|
| DQN | 128 | 0.0005 | 0.99 | 0.005 | eps_decay=0.99 |
| BootstrapDQN | 64 | 0.0005 | 0.99 | 0.005 | mask_prob=0.9, prior_scale=10 |
| SunriseDQN | 64 | 0.0005 | 0.99 | 0.005 | mask_prob=0.9, prior_scale=10, sunrise_temp=20.0 |
| IVDQN | 64 | 0.0005 | 0.99 | 0.005 | mask_prob=0.5, num_nets=5, dynamic_xi=True, minimal_eff_bs=48 |
| MasksembleDQN | 128 | 0.0005 | 0.99 | 0.005 | no_masks=4, scale=2 |

All values are applied automatically from `masure/config.py` when a recognized
`--env` and `--model` combination is passed to `train.py`.

---

## Repository structure

```
DoctorateResearch/
    README.md
    .gitignore
    SANITIZATION_CHECKLIST.md
    HEUQ/
        requirements.txt
        setup.py
        heuq/
            __init__.py
            models.py          six base classifiers (LR, RF, BG, XGB, CB, DNN)
            uncertainty.py     BCR, u_t, u_e, u_a, NLL, BS, BA (Equations 2-6)
            transforms.py      PCATransform, GaussianRandomProjection
            ablation.py        leave-one-out ablation study (Table 6)
            portfolio.py       sub-portfolio selection (Section 5.4)
        demo/
            run_heuq_bank.py   end-to-end demo on UCI Bank Marketing dataset
    MEASURE/
        requirements.txt
        masksembles-main/      vendored Masksembles library (MIT license)
        masure/
            __init__.py
            networks.py        QNetwork, Maskemble, QNet_with_prior
            dqn.py             DQNAgent
            masure_dqn.py      MASUREAgent (core contribution)
            baselines.py       BootstrapDQNAgent, SunriseDQNAgent, IVDQNAgent
            utils.py           ReplayBuffer, MaskReplayBuffer, IV-DQN utilities
            config.py          paper Table II hyperparameters for all 5 agents
            train.py           argparse training entry point
        noisyenv/
            __init__.py
            wrappers.py        StepBurstNoiseObservation, EpisodeAwareEnv
        demo/
            run_masure_lunarlander.py  end-to-end demo on LunarLander-v2
```
---

## License

Each component carries its own license:

- `MEASURE/masksembles-main/`: MIT License (upstream Masksembles, Durasov et al. 2021)
- All other code: see repository root license file.

---

## Acknowledgement

This research was supported by Ethias through the HEC Digital Labs.
