"""Handbook helper: S-, T-, X- and DR-learners on the made-up six-patient example.

The handbook (chapters 03 and 14) uses one tiny teaching dataset: six invented patients with
baseline vision, the drug they got, and their observed letters gained. This script fits each
meta-learner with plain least-squares lines (no regularization, no cross-fitting) so every
number in the handbook's worked example can be checked by hand or reproduced here. The data
are invented; nothing here comes from Protocol T.

Usage:
    python scripts/handbook/six_patients.py
"""

from __future__ import annotations

import numpy as np

VA = np.array([50, 55, 60, 70, 72, 75], float)       # baseline letters
T = np.array([1, 0, 1, 0, 1, 0])                      # 1 = aflibercept, 0 = bevacizumab
Y = np.array([18, 12, 14, 8, 7, 6], float)            # observed letters gained
TRUE_EFFECT = np.array([8, 7, 5, 1, 0, 1], float)     # god's-eye view (chapter 03)
E = 0.5                                               # P(T = 1): 3 of 6 patients


def line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares intercept and slope."""
    A = np.column_stack([np.ones(len(x)), x])
    return np.linalg.solve(A.T @ A, A.T @ y)


def pred(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x


def show(name: str, tau: np.ndarray) -> None:
    pehe = np.sqrt(np.mean((tau - TRUE_EFFECT) ** 2))
    print(f"{name:<11} effects {np.round(tau, 2)}  mean {tau.mean():.2f}  PEHE {pehe:.2f}")


def main() -> None:
    # S-learner: one model Y ~ VA + T; effect = coefficient of T (same for everyone)
    A = np.column_stack([np.ones(6), VA, T])
    s = np.linalg.solve(A.T @ A, A.T @ Y)
    print(f"S-learner fit: Y = {s[0]:.3f} + {s[1]:.4f}*VA + {s[2]:.3f}*T")
    show("S-learner", np.full(6, s[2]))

    # T-learner: one line per arm; effect = mu1(x) - mu0(x)
    b1, b0 = line(VA[T == 1], Y[T == 1]), line(VA[T == 0], Y[T == 0])
    print(f"T-learner fits: mu1 = {b1[0]:.3f} + {b1[1]:.4f}*VA ; "
          f"mu0 = {b0[0]:.3f} + {b0[1]:.4f}*VA")
    mu1, mu0 = pred(b1, VA), pred(b0, VA)
    show("T-learner", mu1 - mu0)

    # X-learner: impute individual effects with the other arm's model, regress them on VA
    d1 = Y[T == 1] - mu0[T == 1]          # treated: observed - predicted control outcome
    d0 = mu1[T == 0] - Y[T == 0]          # controls: predicted treated outcome - observed
    print(f"X-learner imputed effects: treated {np.round(d1, 2)} ; controls {np.round(d0, 2)}")
    t1, t0 = line(VA[T == 1], d1), line(VA[T == 0], d0)
    tau_x = E * pred(t0, VA) + (1 - E) * pred(t1, VA)   # weight g(x) = e(x) on tau0
    show("X-learner", tau_x)

    # DR-learner: doubly robust pseudo-outcome, then regress it on VA
    phi = mu1 - mu0 + T * (Y - mu1) / E - (1 - T) * (Y - mu0) / (1 - E)
    print(f"DR pseudo-outcomes: {np.round(phi, 2)}")
    show("DR-learner", pred(line(VA, phi), VA))

    print(f"True effects {TRUE_EFFECT}  mean {TRUE_EFFECT.mean():.2f}")
    print(f"Difference in means (ATE estimate): "
          f"{Y[T == 1].mean() - Y[T == 0].mean():.2f}")


if __name__ == "__main__":
    main()
