# Finite differences

**What.** We estimate the rate of change (letters per week) from grid values:
- **forward**: (f(t + h) − f(t)) / h, error O(h),
- **backward**: (f(t) − f(t − h)) / h, error O(h),
- **central**: (f(t + h) − f(t − h)) / 2h, error O(h²).

**Why here.** It shows when vision improves fastest (the first weeks of monthly
injections) and how the arms differ in speed (`m7_rates_of_change.png`).

**What it showed** (`m7_finite_difference_orders.md`). On a smooth test function with a
known derivative, halving h halves the forward and backward errors (observed order about
0.95–0.99, theory 1). The central difference's error falls about four times per halving
(order 1.96, theory 2). The theory is confirmed numerically.
