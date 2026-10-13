# Interpolation: linear, cubic spline, Lagrange

**What.** Visits happen at irregular days, but trajectory analysis needs every patient on
the same weekly grid. **Linear interpolation** joins neighbouring visits with straight
lines. A **cubic spline** (`scipy.interpolate`) fits a piecewise cubic with continuous first
and second derivatives, which looks smoother. Neither is allowed to extrapolate beyond a
patient's first or last visit. The result is a 3-D tensor: patients × weeks × (vision, CST,
injected).

**Lagrange interpolation** fits one polynomial of degree n − 1 through all n points. We
implement it from scratch and check it against scipy (coefficients agree to 1e-14,
`m7_lagrange_check.md`).

**Why high degree fails.** Through 5 visits (degree 4) the polynomial spans 9.45 letters,
close to the observed range of 9. Through 12 visits (degree 11) it swings over 45 letters
while the data range is only 13. This is the **Runge phenomenon**: high-degree polynomials
oscillate between nodes, which is why splines (low degree, piecewise) are used instead
(`m7_interpolation.png`).
