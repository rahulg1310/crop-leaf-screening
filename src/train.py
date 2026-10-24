"""
Step 2 - Train the leaf classifier with transfer learning.

    python src/train.py

Idea (transfer learning): MobileNetV2 already learned to see edges, colours
and textures from 1.4 million ImageNet photos. We reuse it in two phases:

  Phase 1 - FREEZE MobileNetV2 and train only a small new final layer that
            maps its features to our 4 leaf classes (fast).
  Phase 2 - FINE-TUNE: unfreeze MobileNetV2's last few layers and train them
            very gently (tiny learning rate) so they specialise in leaf spots.

That is why it trains in about 20 minutes on a normal laptop CPU.
"""
import json
import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import plots as P

tf.keras.utils.set_random_seed(C.SEED)

WEIGHTS_FILE = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
WEIGHTS_MIRROR = ("https://github.com/JonathanCMitchell/mobilenet_v2_keras/"
                  "releases/download/v1.1/" + WEIGHTS_FILE)


def make_dataset(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        C.DATA_DIR / split,
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE,
        shuffle=shuffle,
        seed=C.SEED,
    )


def pretrained_base():
    """Load MobileNetV2 with ImageNet weights (tries Google, then a GitHub mirror)."""
    kwargs = dict(input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3), include_top=False)
    try:
        return tf.keras.applications.MobileNetV2(weights="imagenet", **kwargs)
    except Exception as e:
        print("Default weights download failed (", e, ") - trying GitHub mirror")
        path = tf.keras.utils.get_file(WEIGHTS_FILE, WEIGHTS_MIRROR, cache_subdir="models")
        return tf.keras.applications.MobileNetV2(weights=path, **kwargs)


def build_model(n_classes):
    # Random changes during training so the model copes better with
    # different angles and lighting (they are switched off at prediction time).
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.RandomContrast(0.25),
    ], name="augmentation")

    base = pretrained_base()
    base.trainable = False                     # phase 1: keep the pretrained knowledge

    inputs = tf.keras.Input(shape=(C.IMG_SIZE, C.IMG_SIZE, 3), name="image_0_255")
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="to_minus1_plus1")(x)
    x = base(x, training=False)                # BatchNorm layers stay in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="classes")(x)
    return tf.keras.Model(inputs, outputs, name="tomato_leaf_screener"), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(C.MODEL_PATH, monitor="val_loss",
                                           save_best_only=True),
    ]


def plot_history(h, phase1_epochs):
    epochs = list(range(1, len(h["accuracy"]) + 1))
    marker = phase1_epochs + 0.5
    P.line_plot(epochs, {"Train": h["accuracy"], "Validation": h["val_accuracy"]},
                "Accuracy per epoch", "Epoch", "Accuracy",
                C.RESULTS_DIR / "training_accuracy.png", vline=marker, vlabel="fine-tuning starts")
    P.line_plot(epochs, {"Train": h["loss"], "Validation": h["val_loss"]},
                "Loss per epoch", "Epoch", "Loss",
                C.RESULTS_DIR / "training_loss.png", percent=False,
                vline=marker, vlabel="fine-tuning starts")


def main():
    C.MODEL_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)
    train_ds = make_dataset("train", shuffle=True)
    val_ds = make_dataset("val", shuffle=False)
    class_names = train_ds.class_names
    print("Classes:", class_names)
    C.CLASS_NAMES_PATH.write_text(json.dumps(class_names))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(len(class_names))

    print("\n--- Phase 1: train the new final layer (MobileNetV2 frozen) ---")
    compile_model(model, C.LEARNING_RATE)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=C.EPOCHS, callbacks=callbacks())

    print(f"\n--- Phase 2: fine-tune the last {C.FINE_TUNE_LAYERS} MobileNetV2 layers ---")
    base.trainable = True
    for layer in base.layers[:-C.FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:                  # keep BatchNorm statistics fixed
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, C.FINE_TUNE_LEARNING_RATE)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=C.FINE_TUNE_EPOCHS,
                   callbacks=callbacks())

    model.save(C.MODEL_PATH)
    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    (C.RESULTS_DIR / "training_history.json").write_text(
        json.dumps({"phase1_epochs": len(h1.history["loss"]), **history}, indent=2))
    plot_history(history, len(h1.history["loss"]))
    best = max(history["val_accuracy"])
    print(f"Saved model to {C.MODEL_PATH} (best validation accuracy {best:.1%})")


if __name__ == "__main__":
    main()
