| scenario         | learner                               |   pehe_mean |   pehe_sd |   ate_bias_mean |   heterogeneity_found_rate |   replications |
|:-----------------|:--------------------------------------|------------:|----------:|----------------:|---------------------------:|---------------:|
| a_null           | Causal forest (CausalForestDML)       |        1.8  |     0.609 |         -0.49   |                        0   |             10 |
| a_null           | Constant effect (difference in means) |        1.19 |     0.667 |         -0.214  |                        0   |             10 |
| a_null           | DR-learner (GB)                       |        9.08 |     0.903 |         -0.516  |                        1   |             10 |
| a_null           | DR-learner (Ridge)                    |        8.76 |     0.99  |         -0.38   |                        1   |             10 |
| a_null           | S-learner (GB)                        |        1.92 |     0.735 |         -1.61   |                        0   |             10 |
| a_null           | S-learner (Ridge)                     |        1.43 |     0.895 |         -0.788  |                        0   |             10 |
| a_null           | T-learner (GB)                        |        7.54 |     0.783 |         -0.575  |                        1   |             10 |
| a_null           | T-learner (Ridge)                     |        7.22 |     0.806 |         -0.37   |                        1   |             10 |
| a_null           | TARNet (RMSprop)                      |        4.37 |     1.56  |         -0.494  |                        0.9 |             10 |
| a_null           | X-learner (GB)                        |        6.02 |     0.852 |         -0.546  |                        1   |             10 |
| a_null           | X-learner (Ridge)                     |        6.55 |     0.82  |         -0.429  |                        1   |             10 |
| b_published_like | Causal forest (CausalForestDML)       |        2.8  |     0.689 |          0.0759 |                        0   |             10 |
| b_published_like | Constant effect (difference in means) |        2.88 |     0.59  |         -0.265  |                        0   |             10 |
| b_published_like | DR-learner (GB)                       |        9.36 |     0.809 |         -0.471  |                        1   |             10 |
| b_published_like | DR-learner (Ridge)                    |        8.62 |     0.893 |          0.0427 |                        1   |             10 |
| b_published_like | S-learner (GB)                        |        2.65 |     0.48  |         -1.21   |                        0   |             10 |
| b_published_like | S-learner (Ridge)                     |        2.5  |     0.406 |         -0.0511 |                        0   |             10 |
| b_published_like | T-learner (GB)                        |        7.38 |     0.45  |          0.32   |                        1   |             10 |
| b_published_like | T-learner (Ridge)                     |        7.11 |     1.13  |          0.161  |                        1   |             10 |
| b_published_like | TARNet (RMSprop)                      |        5.28 |     1.76  |         -0.0539 |                        1   |             10 |
| b_published_like | X-learner (GB)                        |        5.91 |     0.496 |          0.28   |                        1   |             10 |
| b_published_like | X-learner (Ridge)                     |        6.47 |     1.08  |          0.163  |                        1   |             10 |
| c_nonlinear      | Causal forest (CausalForestDML)       |        2.44 |     0.624 |          0.0821 |                        0.1 |             10 |
| c_nonlinear      | Constant effect (difference in means) |        2.51 |     0.483 |          0.0147 |                        0   |             10 |
| c_nonlinear      | DR-learner (GB)                       |        9.33 |     0.878 |         -0.262  |                        1   |             10 |
| c_nonlinear      | DR-learner (Ridge)                    |        8.48 |     1.06  |          0.398  |                        1   |             10 |
| c_nonlinear      | S-learner (GB)                        |        2.21 |     0.462 |         -1.05   |                        0   |             10 |
| c_nonlinear      | S-learner (Ridge)                     |        2.3  |     0.483 |         -0.0205 |                        0   |             10 |
| c_nonlinear      | T-learner (GB)                        |        7.5  |     0.552 |         -0.193  |                        1   |             10 |
| c_nonlinear      | T-learner (Ridge)                     |        6.7  |     0.831 |          0.329  |                        1   |             10 |
| c_nonlinear      | TARNet (RMSprop)                      |        4.64 |     0.797 |          0.698  |                        1   |             10 |
| c_nonlinear      | X-learner (GB)                        |        6.09 |     0.605 |         -0.166  |                        1   |             10 |
| c_nonlinear      | X-learner (Ridge)                     |        6.07 |     0.724 |          0.343  |                        1   |             10 |
