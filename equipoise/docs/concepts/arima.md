# Time series: ADF, moving average, AR / ARMA / ARIMA

**What.** We use the arm-mean vision change on a 4-week grid. The **ADF test** asks
whether a series is stationary (has no unit root); if not, we **difference** it.
**Moving-average smoothing** averages 3 neighbouring points. **AR(p)** regresses a value on
its p previous values, **MA(q)** on its q previous errors, and **ARIMA(p, d, q)** does both
after d differences. The order is chosen by **AIC** on weeks 0–52, and we forecast weeks
56–104.

**Honest caveat.** Per-patient series are short and irregular, so this is a
population-level demonstration on 14 points per arm, not a patient forecaster.

**What it showed** (`m7_adf.md`, `m7_arima_forecast.md`). Aflibercept's series is already
stationary in levels (ADF p = 0.014); bevacizumab's is not, even after differencing
(p = 0.059). AIC chose ARIMA(1, 1, 2) for aflibercept and ARIMA(1, 1, 1) for the other two
arms. For weeks 56–104, every ARIMA forecast was **worse** than simply carrying the last
value forward. For example, aflibercept's forecast RMSE was 3.82 letters vs 0.34 for the
naive forecast. Vision plateaus after year 1, and with 14 points ARIMA overfits the
early rise. A legitimate negative result.
