| method       |   trial |   value |   best_so_far | params                                                                                               |
|:-------------|--------:|--------:|--------------:|:-----------------------------------------------------------------------------------------------------|
| Grid search  |       1 |    9.99 |          9.99 | {'learning_rate': 0.02, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 100}                 |
| Grid search  |       2 |    9.93 |          9.93 | {'learning_rate': 0.02, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 300}                 |
| Grid search  |       3 |   10.1  |          9.93 | {'learning_rate': 0.02, 'max_depth': 1, 'min_samples_leaf': 40, 'n_estimators': 100}                 |
| Grid search  |       4 |   10.2  |          9.93 | {'learning_rate': 0.02, 'max_depth': 1, 'min_samples_leaf': 40, 'n_estimators': 300}                 |
| Grid search  |       5 |    9.95 |          9.93 | {'learning_rate': 0.02, 'max_depth': 2, 'min_samples_leaf': 10, 'n_estimators': 100}                 |
| Grid search  |       6 |   10.2  |          9.93 | {'learning_rate': 0.02, 'max_depth': 2, 'min_samples_leaf': 10, 'n_estimators': 300}                 |
| Grid search  |       7 |   10.1  |          9.93 | {'learning_rate': 0.02, 'max_depth': 2, 'min_samples_leaf': 40, 'n_estimators': 100}                 |
| Grid search  |       8 |   10.2  |          9.93 | {'learning_rate': 0.02, 'max_depth': 2, 'min_samples_leaf': 40, 'n_estimators': 300}                 |
| Grid search  |       9 |    9.97 |          9.93 | {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100}                 |
| Grid search  |      10 |   10.3  |          9.93 | {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 300}                 |
| Grid search  |      11 |   10.1  |          9.93 | {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 40, 'n_estimators': 100}                 |
| Grid search  |      12 |   10.2  |          9.93 | {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 40, 'n_estimators': 300}                 |
| Grid search  |      13 |    9.93 |          9.93 | {'learning_rate': 0.05, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 100}                 |
| Grid search  |      14 |   10.3  |          9.93 | {'learning_rate': 0.05, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 300}                 |
| Grid search  |      15 |   10.1  |          9.93 | {'learning_rate': 0.05, 'max_depth': 1, 'min_samples_leaf': 40, 'n_estimators': 100}                 |
| Grid search  |      16 |   10.5  |          9.93 | {'learning_rate': 0.05, 'max_depth': 1, 'min_samples_leaf': 40, 'n_estimators': 300}                 |
| Grid search  |      17 |   10.2  |          9.93 | {'learning_rate': 0.05, 'max_depth': 2, 'min_samples_leaf': 10, 'n_estimators': 100}                 |
| Grid search  |      18 |   10.9  |          9.93 | {'learning_rate': 0.05, 'max_depth': 2, 'min_samples_leaf': 10, 'n_estimators': 300}                 |
| Grid search  |      19 |   10.1  |          9.93 | {'learning_rate': 0.05, 'max_depth': 2, 'min_samples_leaf': 40, 'n_estimators': 100}                 |
| Grid search  |      20 |   10.5  |          9.93 | {'learning_rate': 0.05, 'max_depth': 2, 'min_samples_leaf': 40, 'n_estimators': 300}                 |
| Grid search  |      21 |   10.3  |          9.93 | {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100}                 |
| Grid search  |      22 |   11    |          9.93 | {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 300}                 |
| Grid search  |      23 |   10.1  |          9.93 | {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 40, 'n_estimators': 100}                 |
| Grid search  |      24 |   10.5  |          9.93 | {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 40, 'n_estimators': 300}                 |
| Grid search  |      25 |   10.1  |          9.93 | {'learning_rate': 0.1, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 100}                  |
| Grid search  |      26 |   10.7  |          9.93 | {'learning_rate': 0.1, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 300}                  |
| Grid search  |      27 |   10.3  |          9.93 | {'learning_rate': 0.1, 'max_depth': 1, 'min_samples_leaf': 40, 'n_estimators': 100}                  |
| Grid search  |      28 |   10.8  |          9.93 | {'learning_rate': 0.1, 'max_depth': 1, 'min_samples_leaf': 40, 'n_estimators': 300}                  |
| Grid search  |      29 |   10.6  |          9.93 | {'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 10, 'n_estimators': 100}                  |
| Grid search  |      30 |   11.4  |          9.93 | {'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 10, 'n_estimators': 300}                  |
| Optuna (TPE) |       1 |   10.3  |         10.3  | {'learning_rate': 0.03574712922600244, 'max_depth': 4, 'min_samples_leaf': 45, 'n_estimators': 260}  |
| Optuna (TPE) |       2 |    9.94 |          9.94 | {'learning_rate': 0.01700037298921102, 'max_depth': 1, 'min_samples_leaf': 8, 'n_estimators': 354}   |
| Optuna (TPE) |       3 |   11.1  |          9.94 | {'learning_rate': 0.07725378389307355, 'max_depth': 3, 'min_samples_leaf': 6, 'n_estimators': 390}   |
| Optuna (TPE) |       4 |   10.5  |          9.94 | {'learning_rate': 0.16967533607196555, 'max_depth': 1, 'min_samples_leaf': 15, 'n_estimators': 114}  |
| Optuna (TPE) |       5 |    9.99 |          9.94 | {'learning_rate': 0.028145092716060652, 'max_depth': 3, 'min_samples_leaf': 29, 'n_estimators': 152} |
| Optuna (TPE) |       6 |   10.2  |          9.94 | {'learning_rate': 0.08012737503998542, 'max_depth': 1, 'min_samples_leaf': 21, 'n_estimators': 178}  |
| Optuna (TPE) |       7 |   10.6  |          9.94 | {'learning_rate': 0.04717052037625178, 'max_depth': 4, 'min_samples_leaf': 16, 'n_estimators': 230}  |
| Optuna (TPE) |       8 |   10.3  |          9.94 | {'learning_rate': 0.07500118950416987, 'max_depth': 1, 'min_samples_leaf': 39, 'n_estimators': 109}  |
| Optuna (TPE) |       9 |   10.9  |          9.94 | {'learning_rate': 0.012476394272569451, 'max_depth': 4, 'min_samples_leaf': 59, 'n_estimators': 333} |
| Optuna (TPE) |      10 |   10.2  |          9.94 | {'learning_rate': 0.028180680291847244, 'max_depth': 1, 'min_samples_leaf': 43, 'n_estimators': 204} |
| Optuna (TPE) |      11 |    9.9  |          9.9  | {'learning_rate': 0.011096544503064265, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 287} |
| Optuna (TPE) |      12 |    9.98 |          9.9  | {'learning_rate': 0.01550816873426085, 'max_depth': 1, 'min_samples_leaf': 5, 'n_estimators': 292}   |
| Optuna (TPE) |      13 |    9.86 |          9.86 | {'learning_rate': 0.01183868632641984, 'max_depth': 1, 'min_samples_leaf': 23, 'n_estimators': 343}  |
| Optuna (TPE) |      14 |   10.3  |          9.86 | {'learning_rate': 0.0333156517171016, 'max_depth': 2, 'min_samples_leaf': 37, 'n_estimators': 339}   |
| Optuna (TPE) |      15 |    9.93 |          9.86 | {'learning_rate': 0.01205146407718593, 'max_depth': 2, 'min_samples_leaf': 23, 'n_estimators': 338}  |
| Optuna (TPE) |      16 |    9.89 |          9.86 | {'learning_rate': 0.0102389083067599, 'max_depth': 1, 'min_samples_leaf': 24, 'n_estimators': 271}   |
| Optuna (TPE) |      17 |    9.99 |          9.86 | {'learning_rate': 0.01874627119474705, 'max_depth': 1, 'min_samples_leaf': 34, 'n_estimators': 254}  |
| Optuna (TPE) |      18 |   10.1  |          9.86 | {'learning_rate': 0.027158241592884175, 'max_depth': 1, 'min_samples_leaf': 22, 'n_estimators': 377} |
| Optuna (TPE) |      19 |    9.95 |          9.86 | {'learning_rate': 0.015002905303120383, 'max_depth': 3, 'min_samples_leaf': 19, 'n_estimators': 205} |
| Optuna (TPE) |      20 |   10.3  |          9.86 | {'learning_rate': 0.024248892188235484, 'max_depth': 2, 'min_samples_leaf': 29, 'n_estimators': 395} |
| Optuna (TPE) |      21 |   10.2  |          9.86 | {'learning_rate': 0.0394261112650404, 'max_depth': 1, 'min_samples_leaf': 19, 'n_estimators': 285}   |
| Optuna (TPE) |      22 |   10    |          9.86 | {'learning_rate': 0.019140628213692907, 'max_depth': 2, 'min_samples_leaf': 9, 'n_estimators': 209}  |
| Optuna (TPE) |      23 |    9.95 |          9.86 | {'learning_rate': 0.010050606968760267, 'max_depth': 1, 'min_samples_leaf': 28, 'n_estimators': 213} |
| Optuna (TPE) |      24 |    9.98 |          9.86 | {'learning_rate': 0.010690479936459664, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 187} |
| Optuna (TPE) |      25 |    9.88 |          9.86 | {'learning_rate': 0.011461803318621051, 'max_depth': 1, 'min_samples_leaf': 13, 'n_estimators': 284} |
| Optuna (TPE) |      26 |    9.92 |          9.86 | {'learning_rate': 0.014172200228364212, 'max_depth': 2, 'min_samples_leaf': 23, 'n_estimators': 260} |
| Optuna (TPE) |      27 |   10.3  |          9.86 | {'learning_rate': 0.0263506497694368, 'max_depth': 2, 'min_samples_leaf': 8, 'n_estimators': 287}    |
| Optuna (TPE) |      28 |    9.87 |          9.86 | {'learning_rate': 0.010588123635057624, 'max_depth': 1, 'min_samples_leaf': 23, 'n_estimators': 330} |
| Optuna (TPE) |      29 |   10.1  |          9.86 | {'learning_rate': 0.01039034378766119, 'max_depth': 1, 'min_samples_leaf': 38, 'n_estimators': 331}  |
| Optuna (TPE) |      30 |    9.89 |          9.86 | {'learning_rate': 0.01491245694335196, 'max_depth': 1, 'min_samples_leaf': 27, 'n_estimators': 332}  |
| PSO          |       1 |   10.6  |         10.6  | {'learning_rate': 0.13906791054091117, 'max_depth': 2, 'min_samples_leaf': 52, 'n_estimators': 294}  |
| PSO          |       2 |   10.2  |         10.2  | {'learning_rate': 0.013775626597709352, 'max_depth': 4, 'min_samples_leaf': 47, 'n_estimators': 325} |
| PSO          |       3 |   10    |         10    | {'learning_rate': 0.015461063713132448, 'max_depth': 2, 'min_samples_leaf': 25, 'n_estimators': 374} |
| PSO          |       4 |   10.4  |         10    | {'learning_rate': 0.08934430071955696, 'max_depth': 3, 'min_samples_leaf': 29, 'n_estimators': 130}  |
| PSO          |       5 |   10.6  |         10    | {'learning_rate': 0.06594607868596668, 'max_depth': 1, 'min_samples_leaf': 51, 'n_estimators': 271}  |
| PSO          |       6 |   10.9  |         10    | {'learning_rate': 0.13176119161294714, 'max_depth': 2, 'min_samples_leaf': 58, 'n_estimators': 363}  |
| PSO          |       7 |   10.3  |         10    | {'learning_rate': 0.010000000000000004, 'max_depth': 2, 'min_samples_leaf': 50, 'n_estimators': 306} |
| PSO          |       8 |   10    |         10    | {'learning_rate': 0.012294061461444024, 'max_depth': 2, 'min_samples_leaf': 31, 'n_estimators': 400} |
| PSO          |       9 |   10    |         10    | {'learning_rate': 0.014230513796008138, 'max_depth': 2, 'min_samples_leaf': 25, 'n_estimators': 359} |
| PSO          |      10 |   10.9  |         10    | {'learning_rate': 0.07056376066643093, 'max_depth': 3, 'min_samples_leaf': 23, 'n_estimators': 307}  |
| PSO          |      11 |   10.6  |         10    | {'learning_rate': 0.04506117818047347, 'max_depth': 2, 'min_samples_leaf': 46, 'n_estimators': 370}  |
| PSO          |      12 |   10.5  |         10    | {'learning_rate': 0.03679111879404535, 'max_depth': 2, 'min_samples_leaf': 43, 'n_estimators': 363}  |
| PSO          |      13 |   10.1  |         10    | {'learning_rate': 0.010000000000000004, 'max_depth': 2, 'min_samples_leaf': 45, 'n_estimators': 347} |
| PSO          |      14 |    9.89 |          9.89 | {'learning_rate': 0.011456963128992586, 'max_depth': 1, 'min_samples_leaf': 17, 'n_estimators': 400} |
| PSO          |      15 |    9.98 |          9.89 | {'learning_rate': 0.013427879282213884, 'max_depth': 2, 'min_samples_leaf': 25, 'n_estimators': 349} |
| PSO          |      16 |   10    |          9.89 | {'learning_rate': 0.015331656547385079, 'max_depth': 3, 'min_samples_leaf': 25, 'n_estimators': 380} |
| PSO          |      17 |   10    |          9.89 | {'learning_rate': 0.010000000000000004, 'max_depth': 3, 'min_samples_leaf': 19, 'n_estimators': 400} |
| PSO          |      18 |   10.1  |          9.89 | {'learning_rate': 0.010000000000000004, 'max_depth': 3, 'min_samples_leaf': 7, 'n_estimators': 360}  |
| PSO          |      19 |    9.9  |          9.89 | {'learning_rate': 0.010000000000000004, 'max_depth': 1, 'min_samples_leaf': 29, 'n_estimators': 400} |
| PSO          |      20 |    9.94 |          9.89 | {'learning_rate': 0.010905144139865505, 'max_depth': 1, 'min_samples_leaf': 7, 'n_estimators': 400}  |
| PSO          |      21 |    9.87 |          9.87 | {'learning_rate': 0.011624283594266909, 'max_depth': 1, 'min_samples_leaf': 14, 'n_estimators': 399} |
| PSO          |      22 |    9.89 |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 1, 'min_samples_leaf': 20, 'n_estimators': 400} |
| PSO          |      23 |   10.1  |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 2, 'min_samples_leaf': 5, 'n_estimators': 400}  |
| PSO          |      24 |   10.1  |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 2, 'min_samples_leaf': 5, 'n_estimators': 372}  |
| PSO          |      25 |    9.98 |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 1, 'min_samples_leaf': 5, 'n_estimators': 400}  |
| PSO          |      26 |    9.89 |          9.87 | {'learning_rate': 0.011334942078156288, 'max_depth': 1, 'min_samples_leaf': 10, 'n_estimators': 400} |
| PSO          |      27 |    9.98 |          9.87 | {'learning_rate': 0.01050793396155909, 'max_depth': 1, 'min_samples_leaf': 6, 'n_estimators': 400}   |
| PSO          |      28 |    9.88 |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 1, 'min_samples_leaf': 15, 'n_estimators': 400} |
| PSO          |      29 |    9.89 |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 1, 'min_samples_leaf': 17, 'n_estimators': 400} |
| PSO          |      30 |    9.98 |          9.87 | {'learning_rate': 0.010000000000000004, 'max_depth': 1, 'min_samples_leaf': 5, 'n_estimators': 385}  |
