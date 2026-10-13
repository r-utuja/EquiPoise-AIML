| check                                             |   n |   n_coefficients |   max_abs_diff_vs_sklearn |   condition_number_XtX |   iterations |   converged |   max_abs_diff_vs_statsmodels |
|:--------------------------------------------------|----:|-----------------:|--------------------------:|-----------------------:|-------------:|------------:|------------------------------:|
| OLS normal equations vs sklearn LinearRegression  | 490 |               39 |                  4.62e-14 |                    502 |          nan |         nan |                    nan        |
| IRLS logistic vs statsmodels Logit (gain >= 15)   | 490 |               39 |                nan        |                    nan |            7 |           1 |                      1.26e-15 |
| IRLS logistic vs statsmodels Logit (laser year 1) | 528 |               39 |                nan        |                    nan |            5 |           1 |                      4.44e-16 |
