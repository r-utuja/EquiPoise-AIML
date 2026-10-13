| strategy                | comparison                |   estimate |   se |   ci_low |   ci_high |   n |   df | note                                 |
|:------------------------|:--------------------------|-----------:|-----:|---------:|----------:|----:|-----:|:-------------------------------------|
| complete-case           | Aflibercept - Bevacizumab |       3.95 | 1.07 |   1.85   |      6.06 | 616 |  613 |                                      |
| complete-case           | Ranibizumab - Bevacizumab |       1.82 | 1.08 |  -0.293  |      3.93 | 616 |  613 |                                      |
| kNN (single imputation) | Aflibercept - Bevacizumab |       4.01 | 1    |   2.04   |      5.98 | 660 |  657 | SE ignores imputation uncertainty    |
| kNN (single imputation) | Ranibizumab - Bevacizumab |       2    | 1.01 |   0.0105 |      3.98 | 660 |  657 | SE ignores imputation uncertainty    |
| MICE (m=20, Rubin)      | Aflibercept - Bevacizumab |       3.83 | 1.08 |   1.72   |      5.95 | 660 |  521 | fraction of missing information 0.07 |
| MICE (m=20, Rubin)      | Ranibizumab - Bevacizumab |       1.97 | 1.07 |  -0.125  |      4.06 | 660 |  594 | fraction of missing information 0.04 |
