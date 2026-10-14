| term                                 |     coef |      se |   ci_low |   ci_high |        p | model                     |
|:-------------------------------------|---------:|--------:|---------:|----------:|---------:|:--------------------------|
| Intercept                            | 49.4     | 6.16    |  37.3    |  61.5     | 5.68e-15 | treatment x X_va_letters  |
| treatment[Aflibercept]               | 24.5     | 5.5     |  13.7    |  35.3     | 1.03e-05 | treatment x X_va_letters  |
| treatment[Ranibizumab]               |  6.19    | 5.61    |  -4.83   |  17.2     | 0.271    | treatment x X_va_letters  |
| X_va_letters                         | -0.304   | 0.063   |  -0.428  |  -0.181   | 1.71e-06 | treatment x X_va_letters  |
| treatment[Aflibercept]:X_va_letters  | -0.317   | 0.0832  |  -0.48   |  -0.153   | 0.000156 | treatment x X_va_letters  |
| treatment[Ranibizumab]:X_va_letters  | -0.0697  | 0.085   |  -0.237  |   0.0972  | 0.412    | treatment x X_va_letters  |
| X_cst_um                             | -0.00495 | 0.0033  |  -0.0114 |   0.00153 | 0.134    | treatment x X_va_letters  |
| X_hba1c                              | -0.524   | 0.233   |  -0.981  |  -0.0672  | 0.0246   | treatment x X_va_letters  |
| X_age                                | -0.224   | 0.038   |  -0.299  |  -0.15    | 6.09e-09 | treatment x X_va_letters  |
| X_prior_antivegf                     | -1.66    | 1.13    |  -3.88   |   0.56    | 0.143    | treatment x X_va_letters  |
| Intercept                            | 56.7     | 5.77    |  45.4    |  68.1     | 3.09e-21 | treatment x X_va_below_69 |
| treatment[Aflibercept]               |  1.1     | 1.3     |  -1.45   |   3.65    | 0.398    | treatment x X_va_below_69 |
| treatment[Ranibizumab]               |  1.27    | 1.3     |  -1.29   |   3.82    | 0.331    | treatment x X_va_below_69 |
| X_va_below_69                        | -1.31    | 1.55    |  -4.36   |   1.73    | 0.398    | treatment x X_va_below_69 |
| treatment[Aflibercept]:X_va_below_69 |  5.62    | 1.86    |   1.96   |   9.28    | 0.00269  | treatment x X_va_below_69 |
| treatment[Ranibizumab]:X_va_below_69 |  0.8     | 1.86    |  -2.86   |   4.46    | 0.668    | treatment x X_va_below_69 |
| X_va_letters                         | -0.408   | 0.0529  |  -0.512  |  -0.304   | 4.93e-14 | treatment x X_va_below_69 |
| X_cst_um                             | -0.00469 | 0.00331 |  -0.0112 |   0.00182 | 0.158    | treatment x X_va_below_69 |
| X_hba1c                              | -0.538   | 0.234   |  -0.998  |  -0.0779  | 0.022    | treatment x X_va_below_69 |
| X_age                                | -0.222   | 0.0383  |  -0.298  |  -0.147   | 1.03e-08 | treatment x X_va_below_69 |
| X_prior_antivegf                     | -1.91    | 1.14    |  -4.14   |   0.32    | 0.0931   | treatment x X_va_below_69 |
