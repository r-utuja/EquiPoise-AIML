"""M11 model registry: promote a registered Equipoise bundle version for serving.

M6 registers every real-data bundle as a new version of `equipoise-cate` in the MLflow
model registry. Promotion is a separate, deliberate step: it points the `champion` alias at
one version (MLflow 3 aliases replace the old Staging/Production stages) and records which
learner and training data that version came from.
"""

from __future__ import annotations

from equipoise.causal.bundle import REGISTERED_MODEL, ModelBundle

CHAMPION_ALIAS = "champion"


def promote_bundle(tracking_uri: str, version: str | None = None,
                   alias: str = CHAMPION_ALIAS, bundle: ModelBundle | None = None) -> str:
    """Point the registry alias (default `champion`) at a version of `equipoise-cate`.

    MLflow 3 replaced the old Staging/Production stages with aliases: the served model is
    whichever version carries the alias. Default = the newest version. When the local bundle
    is given, its learner name and training-data hash are recorded as version tags so the
    promoted version can be matched to the file the API serves.
    """
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise LookupError(f"no registered versions of {REGISTERED_MODEL}: run the causal stage")
    chosen = version or str(max(int(v.version) for v in versions))
    if chosen not in {str(v.version) for v in versions}:
        raise LookupError(f"{REGISTERED_MODEL} has no version {chosen}")
    client.set_registered_model_alias(REGISTERED_MODEL, alias, chosen)
    if bundle is not None:
        client.set_model_version_tag(REGISTERED_MODEL, chosen, "effect_model",
                                     bundle.effect_model_name)
        client.set_model_version_tag(REGISTERED_MODEL, chosen, "training_data_hash",
                                     bundle.training_data_hash)
    client.update_model_version(
        REGISTERED_MODEL, chosen,
        description="Equipoise CATE bundle (research prototype, not for clinical use). "
                    f"Alias '{alias}' marks the version the API serves.")
    return chosen
