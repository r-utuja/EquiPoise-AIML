| method       |   trials |   best_cv_rmse | best_params                                                                                          |   default_cv_rmse |
|:-------------|---------:|---------------:|:-----------------------------------------------------------------------------------------------------|------------------:|
| Grid search  |       30 |           9.93 | {'learning_rate': 0.05, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 100}                 |              10.2 |
| Optuna (TPE) |       30 |           9.86 | {'learning_rate': 0.01183868632641984, 'max_depth': 1, 'min_samples_leaf': 23, 'n_estimators': 343}  |              10.2 |
| PSO          |       30 |           9.87 | {'learning_rate': 0.011624283594266909, 'max_depth': 1, 'min_samples_leaf': 14, 'n_estimators': 399} |              10.2 |
