# Newton-Raphson break-even

**What.** Newton-Raphson finds a root of g by repeating x ← x − g(x) / g′(x). The
derivative is either analytic or a central finite difference. We implement it from scratch
and verify every root against scipy's bracketing `brentq`.

**Here.** Value of vision is U(L) = 1 − exp(−kL), a concave curve, so each extra letter is
worth a little less. The net benefit of a drug is W·U(letters) − expected cost.
- The **break-even price** is the aflibercept price at which its net benefit equals
  bevacizumab's. g is linear in price, so Newton lands on the root in one step. This is
  expected: Newton is exact for linear functions.
- The **break-even benefit** is the smallest extra number of letters that would justify
  the extra cost. It is nonlinear, so Newton iterates. When the extra cost exceeds anything
  a benefit could be worth, there is no root, and we report that instead of a number.

**What it showed** (`m8_newton_vs_brentq.md`; 132 held-out patients × 2 drugs). Newton and
brentq agree to 2e-15 (price) and 3e-13 (benefit). Analytic and finite-difference
derivatives agree to 4e-14. Median iterations: 2 for price, 5 for benefit. 5 cases had no
root. With DEMO prices, the median break-even aflibercept price is 3.15 relative units, vs
the demo price of 10 (`m8_population_summary.md`). This is illustrative until the team
fills in real Indian prices.
