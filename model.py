import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


def main():
    print(tf.VERSION)
    print(tf.keras.__version__)

    model = build_model(0.01)

    X_train, y_train = build_data(1000)
    model.fit(X_train, y_train, epochs=10, batch_size=100)

    X_test, y_test = build_data(100)
    print(model.evaluate(X_test, y_test))


def build_data(n=100):
    features = np.random.random((n, 2)) * 10
    # labels must be: n x #labels
    labels = np.array([
            features[:,0] + np.random.random((n,)) * 2 > 0.2,
            features[:,1] + np.random.random((n,)) * 2 < 0.5
    ]).T

    return features, labels


def build_model(learning_rate):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(2,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()

