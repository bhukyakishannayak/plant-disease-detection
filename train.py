"""Plant disease classification using transfer learning (MobileNetV2)
on the PlantVillage dataset (38 classes, ~54k leaf images),
loaded from the Hugging Face Hub (geraldmc/plantvillage-full).

Run:  python train.py
Outputs (in ./outputs): model, class names, training curves,
confusion matrix, and a classification report.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers

IMG_SIZE = 224
BATCH = 32
EPOCHS_HEAD = 5   # train only the new classifier head
EPOCHS_FINE = 3   # then fine-tune the top of MobileNetV2
AUTOTUNE = tf.data.AUTOTUNE
DATASET = "geraldmc/plantvillage-full"  # PlantVillage with a leaf-grouped train/test split

os.makedirs("outputs", exist_ok=True)


def hf_to_tf(hf_ds, label_col, label_to_id):
    """Wrap a Hugging Face dataset as a tf.data.Dataset of (image, label)."""

    def gen():
        for ex in hf_ds:
            img = ex["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            yield np.asarray(img, dtype=np.uint8), label_to_id(ex[label_col])

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.uint8),
            tf.TensorSpec((), tf.int32),
        ),
    )


def load_data():
    """Downloads PlantVillage from the Hugging Face Hub with a leaf-grouped split.

    All images of the same physical leaf stay in the same split, so the test
    accuracy is not inflated by near-duplicate leaves. The validation set is
    carved out of the training set the same way (grouped by leaf_id).
    """
    ds = load_dataset(DATASET)
    print(ds)
    first = next(iter(ds.values())) if isinstance(ds, DatasetDict) else ds

    if "split" in first.column_names:       # one table with a 'split' column
        full = concatenate_datasets(list(ds.values())) if isinstance(ds, DatasetDict) else ds
        train_full = full.filter(lambda s: s == "train", input_columns="split")
        test_hf = full.filter(lambda s: s == "test", input_columns="split")
    else:                                   # ready-made train/test splits
        train_full, test_hf = ds["train"], ds["test"]
    print("Columns:", train_full.column_names)

    # Which column holds the class label?
    label_col = "class_label" if "class_label" in train_full.column_names else "label"
    feat = train_full.features[label_col]
    if hasattr(feat, "names"):              # ClassLabel: labels are already ints
        class_names = list(feat.names)
        label_to_id = int
    else:                                   # labels are strings
        class_names = sorted(set(train_full[label_col]))
        label_to_id = {n: i for i, n in enumerate(class_names)}.__getitem__
    print(f"{len(class_names)} classes")

    # Validation split: 10% of leaves (not of images) from the training set
    if "leaf_id" in train_full.column_names:
        rng = np.random.default_rng(42)
        leaf_ids = np.unique(np.array(train_full["leaf_id"]))
        rng.shuffle(leaf_ids)
        val_ids = set(leaf_ids[: int(0.1 * len(leaf_ids))].tolist())
        train_hf = train_full.filter(lambda x: x not in val_ids, input_columns="leaf_id")
        val_hf = train_full.filter(lambda x: x in val_ids, input_columns="leaf_id")
    else:
        parts = train_full.train_test_split(test_size=0.1, seed=42)
        train_hf, val_hf = parts["train"], parts["test"]
    print(f"train={len(train_hf)}  val={len(val_hf)}  test={len(test_hf)}")

    return (
        hf_to_tf(train_hf, label_col, label_to_id),
        hf_to_tf(val_hf, label_col, label_to_id),
        hf_to_tf(test_hf, label_col, label_to_id),
        class_names,
    )


def make_pipeline(ds, training=False):
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32), lbl), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


def build_model(num_classes):
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)                              # only active during training
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def plot_history(histories):
    acc = sum([h.history["accuracy"] for h in histories], [])
    val_acc = sum([h.history["val_accuracy"] for h in histories], [])
    loss = sum([h.history["loss"] for h in histories], [])
    val_loss = sum([h.history["val_loss"] for h in histories], [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="validation")
    axes[0].axvline(EPOCHS_HEAD - 1, color="gray", ls="--", label="fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    plt.close()


def evaluate(model, test_ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0,
    )
    print(report)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.colorbar()
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()


def main():
    train_raw, val_raw, test_raw, class_names = load_data()
    print(f"{len(class_names)} classes found")

    train_ds = make_pipeline(train_raw, training=True)
    val_ds = make_pipeline(val_raw)
    test_ds = make_pipeline(test_raw)

    model, base = build_model(len(class_names))

    # Stage 1: train the classifier head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # Stage 2: fine-tune the last 30 layers with a small learning rate
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

    plot_history([h1, h2])

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    evaluate(model, test_ds, class_names)

    model.save("outputs/plant_disease_model.keras")
    with open("outputs/class_names.json", "w") as f:
        json.dump(class_names, f)
    print("Saved model and class names to ./outputs")


if __name__ == "__main__":
    main()
