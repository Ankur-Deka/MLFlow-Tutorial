import mlflow
import os

with mlflow.start_run(run_name="hellow_mlflow"):
    # log any parameter like learning rate (any number)
    lr = 1.5e-4
    mlflow.log_param("lr", lr)

    # training epochs
    for e in range(10):
        # log metrics over time (any number)
        dummy_accuracy = 1 - 1 / ((1e4 * lr)**e)
        mlflow.log_metric("accuracy", dummy_accuracy)

        # log checkpoint as artifact (artifact is any file or directory)
        os.system("echo I am a dummy checkpoint > dummy_checkpoint.ckpt")
        mlflow.log_artifact("dummy_checkpoint.ckpt")
