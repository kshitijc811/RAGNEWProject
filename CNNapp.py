from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tensorflow.keras.utils import to_categorical

from cnn_model import build_cnn


CIFAR_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

st.set_page_config(
    page_title="CNN image classifier",
    page_icon=":material/photo_camera:",
    layout="wide",
)
st.title("CNN image classifier")
st.caption("Train a small convolutional neural network on CIFAR-10 or your own image folders.")

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.img_size = 32
    st.session_state.class_names = []


@st.cache_data
def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (
        x_train.astype("float32") / 255.0,
        to_categorical(y_train, 10),
        x_test.astype("float32") / 255.0,
        to_categorical(y_test, 10),
    )


def show_history(history) -> None:
    chart_data = {
        "train accuracy": history.history.get("accuracy", []),
        "validation accuracy": history.history.get("val_accuracy", []),
        "train loss": history.history.get("loss", []),
        "validation loss": history.history.get("val_loss", []),
    }
    st.line_chart(chart_data, x_label="Epoch", y_label="Value")


def safe_extract_zip(uploaded_zip, destination: Path) -> None:
    with zipfile.ZipFile(uploaded_zip) as archive:
        destination_root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError("The ZIP contains an unsafe path.")
        archive.extractall(destination)


with st.sidebar:
    st.header("Controls")
    section = st.radio("Section", ["Train model", "Predict"])

if section == "Train model":
    dataset_choice = st.radio("Dataset", ["CIFAR-10", "Custom image folders"])
    img_size = st.slider("Image size", 32, 128, 64, step=8)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    epochs = st.slider("Epochs", 1, 20, 5)

    if dataset_choice == "CIFAR-10":
        st.subheader("CIFAR-10")
        st.write("The built-in dataset contains 10 common object classes.")
        if st.button("Train on CIFAR-10", type="primary", icon=":material/model_training:"):
            with st.spinner("Loading CIFAR-10 and training the model..."):
                x_train, y_train, x_test, y_test = load_cifar10_data()
                model = build_cnn((32, 32, 3), 10)
                history = model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
            st.session_state.model = model
            st.session_state.img_size = 32
            st.session_state.class_names = CIFAR_LABELS
            st.success("Training complete.")
            show_history(history)
    else:
        st.subheader("Custom image folders")
        st.info("Upload a ZIP containing one folder per class, for example `cats/` and `dogs/`.")
        zip_file = st.file_uploader("Dataset ZIP", type=["zip"])
        if zip_file:
            temp_dir = Path(tempfile.mkdtemp(prefix="cnn_dataset_"))
            try:
                safe_extract_zip(zip_file, temp_dir)
                data_root = next(
                    (path for path in [temp_dir, *temp_dir.iterdir()] if path.is_dir()
                     and sum(child.is_dir() for child in path.iterdir()) >= 2),
                    None,
                )
                if data_root is None:
                    raise ValueError("ZIP must contain at least two class folders.")
                datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
                train_gen = datagen.flow_from_directory(
                    data_root,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode="categorical",
                    subset="training",
                    shuffle=True,
                )
                val_gen = datagen.flow_from_directory(
                    data_root,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode="categorical",
                    subset="validation",
                    shuffle=False,
                )
                if train_gen.num_classes < 2:
                    raise ValueError("The dataset must contain at least two classes.")
                st.success(f"Found {train_gen.num_classes} classes and {train_gen.samples} training images.")
                if st.button("Train on custom dataset", type="primary", icon=":material/model_training:"):
                    with st.spinner("Training the model..."):
                        model = build_cnn((img_size, img_size, 3), train_gen.num_classes)
                        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=0)
                    st.session_state.model = model
                    st.session_state.img_size = img_size
                    st.session_state.class_names = [
                        name for name, _ in sorted(train_gen.class_indices.items(), key=lambda item: item[1])
                    ]
                    st.success("Training complete.")
                    show_history(history)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                st.error(f"Could not load the dataset: {error}", icon=":material/error:")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

else:
    st.subheader("Predict an image")
    uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.session_state.model is not None:
        img = load_img(
            uploaded_file,
            target_size=(st.session_state.img_size, st.session_state.img_size),
        )
        st.image(img, caption="Uploaded image", width="content")
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        prediction = st.session_state.model.predict(img_array, verbose=0)[0]
        best_index = int(np.argmax(prediction))
        st.success(
            f"Predicted: {st.session_state.class_names[best_index]} "
            f"({prediction[best_index] * 100:.2f}% confidence)"
        )
    elif uploaded_file:
        st.warning("Train a model first in the Train model section.")
    else:
        st.info("Train a model, then upload an image here.")
