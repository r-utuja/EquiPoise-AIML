| scenario         | learner                               |   pehe_mean |   ate_bias_mean |
|:-----------------|:--------------------------------------|------------:|----------------:|
| a_null           | Cluster x arm means (k-means)         |        2.3  |         -0.238  |
| b_published_like | Cluster x arm means (k-means)         |        3.59 |         -0.0992 |
| c_nonlinear      | Cluster x arm means (k-means)         |        3.18 |         -0.139  |
| a_null           | Causal forest (CausalForestDML)       |        1.8  |         -0.49   |
| a_null           | Constant effect (difference in means) |        1.19 |         -0.214  |
| b_published_like | Causal forest (CausalForestDML)       |        2.8  |          0.0759 |
| b_published_like | Constant effect (difference in means) |        2.88 |         -0.265  |
| c_nonlinear      | Causal forest (CausalForestDML)       |        2.44 |          0.0821 |
| c_nonlinear      | Constant effect (difference in means) |        2.51 |          0.0147 |
