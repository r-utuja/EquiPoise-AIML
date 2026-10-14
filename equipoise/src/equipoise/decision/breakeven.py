"""Break-even price and break-even benefit, solved with Newton-Raphson written from scratch.

Value of vision: U(L) = 1 - exp(-k L) for L letters gained - concave, so each extra letter
is worth a little less than the previous one. With willingness to pay W per unit of U, the
net benefit of a drug is W * U(letters) - expected cost. Newton-Raphson finds a root of g by
repeating x <- x - g(x) / g'(x); the derivative is either analytic or a central finite
difference. We verify every root against scipy's bracketing brentq.

- Break-even price: the aflibercept price at which its net benefit equals bevacizumab's.
  With this cost model g(price) is linear in price, so Newton lands on the root in one step
  (that is expected, not a bug: Newton is exact for linear functions).
- Break-even benefit: the smallest extra letters (vs bevacizumab) that would make the
  dearer drug worth its extra cost at current prices. Here g is nonlinear (U saturates), so
  Newton iterates; if the extra cost exceeds everything any benefit could be worth, there is
  no root and we say so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class NewtonResult:
    root: float | None
    iterations: int
    converged: bool
    path: list[float]
    message: str


def value_of_vision(letters: np.ndarray | float, k: float) -> np.ndarray | float:
    return 1.0 - np.exp(-k * np.asarray(letters, float))


def newton(g: Callable[[float], float], x0: float, dg: Callable[[float], float] | None = None,
           tol: float = 1e-10, max_iter: int = 50, fd_step: float = 1e-4) -> NewtonResult:
    """Newton-Raphson for g(x) = 0. `dg=None` uses a central finite difference."""
    x = float(x0)
    path = [x]
    for it in range(1, max_iter + 1):
        gx = g(x)
        d = dg(x) if dg is not None else (g(x + fd_step) - g(x - fd_step)) / (2 * fd_step)
        if not np.isfinite(d) or abs(d) < 1e-14:
            return NewtonResult(None, it, False, path, "derivative is zero: no Newton step")
        x_new = x - gx / d
        path.append(x_new)
        if not np.isfinite(x_new):
            return NewtonResult(None, it, False, path, "diverged")
        if abs(x_new - x) < tol * max(1.0, abs(x)):
            return NewtonResult(x_new, it, True, path, "converged")
        x = x_new
    return NewtonResult(None, max_iter, False, path, "did not converge")


def brent_root(g: Callable[[float], float], lo: float, hi: float) -> float | None:
    """scipy brentq on [lo, hi]; None when g has no sign change there (no root)."""
    if np.sign(g(lo)) == np.sign(g(hi)):
        return None
    return float(brentq(g, lo, hi, xtol=1e-12))


@dataclass
class BreakEven:
    price: float | None              # break-even price per injection of the dearer drug
    price_newton: NewtonResult
    price_newton_fd: NewtonResult
    price_brentq: float | None
    benefit_letters: float | None    # minimum extra letters needed at current prices
    benefit_newton: NewtonResult
    benefit_newton_fd: NewtonResult
    benefit_brentq: float | None
    message: str


def break_even(letters_ref: float, letters_drug: float, inj_ref: float, inj_drug: float,
               laser_ref: float, laser_drug: float, price_ref: float, price_drug: float,
               laser_price: float, wtp: float, k: float, tol: float = 1e-10,
               max_iter: int = 50, fd_step: float = 1e-4) -> BreakEven:
    """Break-even price of the dearer drug and break-even benefit, for one patient."""
    cost_ref = inj_ref * price_ref + laser_ref * laser_price
    gain_value = wtp * (value_of_vision(letters_drug, k) - value_of_vision(letters_ref, k))

    def g_price(p: float) -> float:          # net benefit difference as a function of price
        return float(gain_value - (inj_drug * p + laser_drug * laser_price - cost_ref))

    def dg_price(p: float) -> float:
        return float(-inj_drug)

    pn = newton(g_price, price_drug, dg_price, tol, max_iter, fd_step)
    pfd = newton(g_price, price_drug, None, tol, max_iter, fd_step)
    upper = max(10 * price_drug, 1.0)
    pb = brent_root(g_price, 0.0, upper * 1e3)
    price = pn.root if pn.root is not None and pn.root >= 0 else None
    msg = []
    if pn.root is not None and pn.root < 0:
        msg.append("no break-even price >= 0: the drug is not worth its other costs even free")
        pn.message = msg[-1]

    d_cost = (inj_drug * price_drug + laser_drug * laser_price) - cost_ref

    def g_benefit(delta: float) -> float:    # value of an extra `delta` letters minus extra cost
        return float(wtp * (value_of_vision(letters_ref + delta, k)
                            - value_of_vision(letters_ref, k)) - d_cost)

    def dg_benefit(delta: float) -> float:
        return float(wtp * k * np.exp(-k * (letters_ref + delta)))

    ceiling = wtp * np.exp(-k * letters_ref)   # limit of g_benefit + d_cost as delta -> inf
    if d_cost <= 0:
        bn = bfd = NewtonResult(0.0, 0, True, [0.0], "dearer drug is not dearer: any benefit")
        bb = 0.0
    elif d_cost >= ceiling:
        bn = bfd = NewtonResult(None, 0, False, [], "no root: extra cost exceeds what any "
                                "benefit is worth at this willingness to pay")
        bb = None
    else:
        exact = -np.log(1 - d_cost / ceiling) / k   # used only to bracket brentq
        bn = newton(g_benefit, 5.0, dg_benefit, tol, max_iter, fd_step)
        bfd = newton(g_benefit, 5.0, None, tol, max_iter, fd_step)
        bb = brent_root(g_benefit, 0.0, 2 * exact + 1)
    return BreakEven(price, pn, pfd, pb, bn.root, bn, bfd, bb, "; ".join(msg) or "ok")
