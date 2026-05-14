import mlflow
from mlflow.tracking import MlflowClient


def evaluate_and_promote():
    client = MlflowClient()
    model_name = "HeartDiseaseModel"

    # Get the performance of the current 'production' model
    prod_version = client.get_model_version_by_alias(model_name, "production")
    prod_run_id = prod_version.run_id
    prod_accuracy = client.get_run(prod_run_id).data.metrics['accuracy']

    # Get accuracy of the latest run [cite: 67]
    latest_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1).iloc[0]
    new_accuracy = latest_run['metrics.accuracy']

    # Compare and Promote [cite: 64, 75]
    if new_accuracy > prod_accuracy:
        client.set_registered_model_alias(model_name, "production", latest_run.run_id)
        print("New model promoted to Production!")
    else:
        print("New model did not outperform Production. Keeping old model.")