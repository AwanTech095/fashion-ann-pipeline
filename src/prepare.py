from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save(raw_dir / "x_train.npy", x_train)
    np.save(raw_dir / "y_train.npy", y_train)
    np.save(raw_dir / "x_test.npy", x_test)
    np.save(raw_dir / "y_test.npy", y_test)

    print("Raw Fashion-MNIST data saved to data/raw/")
    print(f"Training images: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel range: {x_train.min()} to {x_train.max()}")


if __name__ == "__main__":
    main()