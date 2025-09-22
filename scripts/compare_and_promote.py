import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Initialize MLflow client
client = MlflowClient()

# Get experiment by name
experiment_name = "Phishing Detection Models"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Get all runs sorted by F1-score
runs = client.search_runs(experiment_ids=experiment_id, order_by=["metrics.f1_score DESC"])

# Collect key metrics for each run
records = []
for run in runs:
    run_data = run.data
    run_id = run.info.run_id
    model_param = run_data.params.get("model", None)

    # Only proceed if model name is valid
    if model_param:
        records.append({
            "Run ID": run_id,
            "Model": model_param,
            "Accuracy": run_data.metrics.get("accuracy", 0),
            "Precision": run_data.metrics.get("precision", 0),
            "Recall": run_data.metrics.get("recall", 0),
            "F1-score": run_data.metrics.get("f1_score", 0),
        })

# Create and save DataFrame
df_metrics = pd.DataFrame(records)
print("\n=== Model Comparison Report ===")
print(df_metrics.sort_values("F1_score", ascending=False))
df_metrics.to_csv("reports/model_comparison_report.csv", index=False)

# Select best model
best_run = df_metrics.sort_values("F1_score", ascending=False).iloc[0]
best_run_id = best_run["Run ID"]
model_name = best_run["Model"]

# Register best model
print(f"\nRegistering model: {model_name} from Run ID: {best_run_id}")
registered_model = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=model_name
)

# Promote best version to Production
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"\nModel '{model_name}' version {registered_model.version} promoted to Production.")
