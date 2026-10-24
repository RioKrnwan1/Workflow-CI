import os
import argparse
import shutil

import mlflow
import mlflow.sklearn

from PIL import Image
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_dataset(dataset_path, image_size=(32, 32)):
    features = []
    labels = []

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {dataset_path}"
        )

    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, filename)

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB").resize(image_size)
                    pixels = list(image.getdata())

                    features.append([
                        channel / 255.0
                        for pixel in pixels
                        for channel in pixel
                    ])
                    labels.append(class_name)

            except Exception as error:
                print(f"Melewati {image_path}: {error}")

    if not features:
        raise ValueError(
            f"Tidak ada gambar yang berhasil dibaca: {dataset_path}"
        )

    return features, labels


def train_and_evaluate(dataset_path):
    X, labels = load_dataset(dataset_path)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    print(f"Jumlah gambar: {len(X)}")
    print(f"Jumlah kelas: {len(encoder.classes_)}")
    print(f"Kelas: {list(encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC())
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [1, 10],
            "classifier__kernel": ["linear", "rbf"]
        },
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    predictions = search.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=[str(name) for name in encoder.classes_],
        zero_division=0
    )

    print(f"\nBest parameters: {search.best_params_}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)

    return search, accuracy, macro_f1, report


def log_results(run_id, search, accuracy, macro_f1, report):
    client = MlflowClient()

    for key, value in search.best_params_.items():
        client.log_param(run_id, key, value)

    client.log_metric(run_id, "test_accuracy", accuracy)
    client.log_metric(run_id, "test_macro_f1", macro_f1)

    os.makedirs("evaluation", exist_ok=True)
    report_path = os.path.join(
        "evaluation",
        "classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    model_dir = "model_artifact"

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Tetapkan Python dan dependensi model secara eksplisit.
    model_conda_env = {
        "name": "plant-disease-model-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.12.7",
            "pip",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "scikit-learn==1.9.1",
                    "pillow",
                    "cloudpickle"
                ]
            }
        ]
    }

    mlflow.sklearn.save_model(
        sk_model=search.best_estimator_,
        path=model_dir,
        conda_env=model_conda_env
    )

    # Simpan isi model pada artifact path "model".
    client.log_artifacts(
        run_id,
        model_dir,
        artifact_path="model"
    )

    client.log_artifact(run_id, report_path)

    print(f"\nMLflow run ID: {run_id}")
    print("Parameter, metrik, model, dan laporan berhasil dicatat.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_preprocessing/images"
    )
    args = parser.parse_args()

    mlflow.set_experiment("Workflow CI Plant Disease")

    run_id = os.environ.get("MLFLOW_RUN_ID")

    search, accuracy, macro_f1, report = train_and_evaluate(
        args.dataset
    )

    if run_id:
        log_results(
            run_id,
            search,
            accuracy,
            macro_f1,
            report
        )
    else:
        with mlflow.start_run() as run:
            log_results(
                run.info.run_id,
                search,
                accuracy,
                macro_f1,
                report
            )


if __name__ == "__main__":
    main()