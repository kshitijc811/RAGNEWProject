from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> Sequential:
    if len(input_shape) != 3 or min(input_shape) <= 0:
        raise ValueError("input_shape must contain positive height, width, and channels.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
