import json
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_FILE = "data.json"


def plotHistory(history):
    fig, ax = plt.subplots(2)

    # Plot Train Accuracy against Test Accuracy
    ax[0].plot(history.history["accuracy"], label="train accuracy")
    ax[0].plot(history.history["val_accuracy"], label="test accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(loc="lower right")
    ax[0].set_title("Accuracy eval")

    # Plot Train Loss against Test Loss
    ax[1].plot(history.history["loss"], label="train error")
    ax[1].plot(history.history["val_loss"], label="test error")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(loc="lower right")
    ax[1].set_title("Loss eval")
    plt.show()


def loadData(dataset):
    with open(dataset, "r") as fp:
        data = json.load(fp)
    inputs = np.array(data["mfcc"])
    targets = np.array(data['labels'])
    return inputs, targets


def buildModel(input_shape):
    model = keras.Sequential([
        # 1st conv
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPool2D((3, 3), (2, 2), padding="same"),
        keras.layers.BatchNormalization(),

        # 2nd conv
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPool2D((3, 3), (2, 2), padding="same"),
        keras.layers.BatchNormalization(),

        # 3rd conv
        keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape),
        keras.layers.MaxPool2D((2, 2), (2, 2), padding="same"),
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        # Dense Layer
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),

        # Output Layer
        keras.layers.Dense(10, activation="softmax")
    ])
    return model


def prepareDatasets(inputs, targets):
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.25)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

    # Add 3d Axis for convolution layers
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x_train, x_test, x_validation, y_train, y_test, y_validation


if __name__ == "__main__":
    # Load Data
    inputs, targets = loadData(DATA_FILE)

    # Prepare Datasets
    x_train, x_test, x_validation, y_train, y_test, y_validation = prepareDatasets(inputs, targets)

    # Build Model
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = buildModel(input_shape)

    # Compile Model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train Model
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=30, batch_size=32)

    # Evaluate
    plotHistory(history)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy = ", test_accuracy, " Error=  ", test_error)
