import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    with open("params.yaml", encoding="utf-8-sig") as file:
        params = yaml.safe_load(file)["train"]

    processed_dir = Path("data/processed")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    x_test = np.load(processed_dir / "x_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model = tf.keras.models.load_model(
        "models/model.h5",
        compile=False,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=params["batch_size"],
        verbose=0,
    )

    probabilities = model.predict(
        x_test,
        batch_size=params["batch_size"],
        verbose=0,
    )
    predictions = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Fashion-MNIST Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_samples": int(len(y_test)),
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test samples: {len(y_test)}")
    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to reports/confusion_matrix.png")
    print(
        "85% accuracy target: "
        + ("MET" if test_accuracy >= 0.85 else "NOT MET")
    )


if __name__ == "__main__":
    main()