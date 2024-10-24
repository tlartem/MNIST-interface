import tensorflow as tf

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    return (x_train, y_train), (x_test, y_test)

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = create_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    model.save("my_model.keras")
    print("Модель сохранена как 'my_model.keras'")

if __name__ == "__main__":
    main()
