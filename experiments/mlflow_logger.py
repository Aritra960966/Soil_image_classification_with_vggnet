import mlflow
import mlflow.tensorflow
import config

mlflow.set_tracking_uri(config.MLFLOW_URI)
mlflow.set_experiment("Soil Classification")

def log_params():
    mlflow.log_param("image_size", config.IMAGE_SIZE)
    mlflow.log_param("batch_size", config.BATCH_SIZE)
    mlflow.log_param("epochs", config.EPOCHS)
    mlflow.log_param("learning_rate", config.LEARNING_RATE)

def log_metrics(history):
    for epoch in range(config.EPOCHS):
        mlflow.log_metric("accuracy", history.history['acc'][epoch], step=epoch)

def log_model(model):
    mlflow.tensorflow.log_model(model, artifact_path="model")
