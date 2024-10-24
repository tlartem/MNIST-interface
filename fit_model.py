import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=15,
        validation_data=(x_test, y_test),
        verbose=1
    )

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = create_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    model.save("my_model.keras")
    print("Улучшенная модель сохранена как 'my_model.keras'")

    # Оценка модели
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точность на тестовом наборе: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
