from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


def main():
    with open("params.yaml", encoding="utf-8-sig") as file:
        params = yaml.safe_load(file)["train"]

    tf.keras.utils.set_random_seed(params["seed"])
    tf.config.experimental.enable_op_determinism()

    processed_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(processed_dir / "x_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    x_val = np.load(processed_dir / "x_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=x_train.shape[1:]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            params["dense_units"],
            activation="relu",
        ),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"],
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[
            tf.keras.callbacks.CSVLogger(
                str(models_dir / "history.csv"),
                append=False,
            ),
        ],
        verbose=2,
    )

    model.save(str(models_dir / "model.h5"))

    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()