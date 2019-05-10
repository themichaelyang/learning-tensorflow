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
    X_val, y_val = build_data(100)
    model.fit(X_train, y_train, epochs=100, batch_size=100, validation_data=(X_val, y_val))

    X_test, y_test = build_data(100)

    fig = plot_boundary(model, X_test, y_test)

    plt.show()


def plot_boundary(model, features_test, labels_test):
    # different x, y than label y! these is feature space
    color_map = plt.get_cmap('coolwarm')
    x_span = np.linspace(0, 10, 100)
    y_span = x_span

    grid = np.meshgrid(x_span, y_span)
    x_grid = grid[0].reshape((-1,))
    y_grid = grid[1].reshape((-1,))

    predictions = model.predict(np.array([x_grid, y_grid]).T)
    fig, ax = plt.subplots()
    ax.tricontourf(x_grid, y_grid, np.argmax(predictions, axis=1), cmap=color_map, alpha=0.25)

    x_feats = features_test[:,0]
    y_feats = features_test[:,1]
    ax.scatter(x_feats, y_feats, c=np.argmax(labels_test, axis=1), cmap=color_map, s=2)

    return fig


def build_data(n=100):
    features = np.random.random((n, 2)) * 10
    # labels must be: n x #labels
    labels = np.array([
            features[:,0] + np.random.random((n,)) * 2 > 5.2,
            features[:,1] + np.random.random((n,)) * 2 < 7.5
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

