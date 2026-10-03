from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", encoding="utf-8-sig") as file:
        params = yaml.safe_load(file)["preprocess"]

    if not 0 < params["test_size"] < 1:
        raise ValueError("preprocess.test_size must be between 0 and 1")
    

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    x_train = np.load(raw_dir / "x_train.npy")
    y_train = np.load(raw_dir / "y_train.npy")
    x_test = np.load(raw_dir / "x_test.npy")
    y_test = np.load(raw_dir / "y_test.npy")

    # Convert raw pixel intensities to the range [0, 1].
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    for name, array in arrays.items():
        np.save(processed_dir / f"{name}.npy", array)

    print("Processed data saved to data/processed/")
    for split in ("train", "val", "test"):
        images = arrays[f"x_{split}"]
        labels = arrays[f"y_{split}"]
        print(f"{split}: images={images.shape}, labels={labels.shape}")
        print(
            f"  dtype={images.dtype}, "
            f"pixel range={images.min():.1f} to {images.max():.1f}"
        )


if __name__ == "__main__":
    main()