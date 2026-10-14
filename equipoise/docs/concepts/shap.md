# SHAP and subgroup checks

**What.** SHAP splits one prediction into additive contributions, one per feature. The
start is the average prediction over background patients; each feature then pushes this
patient's prediction up or down, and the pushes sum exactly to the prediction. We explain
the *effect* (aflibercept − bevacizumab) with the model-agnostic permutation explainer and
sum one-hot columns back to their original feature. In the app, the background is 10
k-means centroids of training patients, so no patient rows are shipped.

**What it showed** (`m9_shap_importance_aflibercept.md`). Additivity holds to 1e-6. The top
drivers of the predicted aflibercept advantage are CST (mean |SHAP| 0.64 letters) and
baseline vision (0.43), followed by a long tail of small contributions. All are a fraction
of a letter, consistent with weak per-patient heterogeneity.

**Subgroups / fairness** (`m9_subgroups.md`). We compare outcome-model error and effect
calibration by sex, race, diabetes type and vision band, using only predictions from
models that did not see the patient.
- Letters RMSE is higher for women (10.7) than men (9.0), and for Black patients (12.7, n =
  95). Their outcomes vary more.
- "Diabetes type: uncertain" (n = 12) is flagged **unreliable** and not interpreted.
- The model under-predicts the observed aflibercept effect in the vision < 69 group
  (predicted 3.57 vs observed 7.24 letters): the forest shrinks effects toward the average.
