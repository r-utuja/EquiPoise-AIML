# Preprocessing: harmonization, outliers, encoding, discretization

**CST harmonization.** OCT machines measure retinal thickness differently: Spectralis and
Cirrus read thicker than Stratus. We implement two fixes. The first is a **median offset**
to Stratus-equivalent units (+43 µm Cirrus, +67 µm Spectralis; Bressler et al., JAMA
Ophthalmol 2014). The second, and the default, is **within-machine z-scoring**: subtract
each machine's training mean and divide by its SD. Published linear conversion equations
can be added in `config/default.yaml` once the team has the coefficients with a citation.
In `m1_cst_harmonization.md`, raw means are 433 / 453 / 467 µm (Stratus / Cirrus /
Spectralis). After z-scoring every machine has mean ≈ 0 and SD ≈ 1.

**Outliers.** Three views of "unusual patient":
- the **IQR rule** (a value beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR, in at least 2 features),
- **Isolation Forest** (points that random splits isolate quickly),
- **DBSCAN** noise points (not inside any dense region; eps from the k-distance curve).

They flag 17, 38 and 5 patients. Only 2 are flagged by all three
(`m1_outlier_overlap.md`), which shows each method has a different idea of "outlier".
Outliers are flagged, not dropped: in a randomized trial, dropping patients could unbalance
the arms.

**Encoding and scaling.** One-hot encoding for nominal features, stored as `scipy.sparse`
(see `docs/data_structures.md`). Ordinal codes for DR severity, standard scaling, and `log`
for right-skewed labs (HbA1c, BMI). Everything is one `ColumnTransformer` from
`build_preprocessor(config)`, fitted on training folds only, so nothing about test patients
leaks into the scaling or imputation.

**Discretization.** HbA1c, vision and age bands (config `preprocess.bands`) turn numbers
into items for the Apriori rules in M3.
