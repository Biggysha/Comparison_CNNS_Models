import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from utils import build_lenet, build_alexnet, build_vggnet, build_googlenet, build_resnet
import os

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define models
models = {
    'lenet': build_lenet(),
    'alexnet': build_alexnet(),
    'vggnet': build_vggnet(),
    'googlenet': build_googlenet(),
    'resnet': build_resnet()
}

# Train and save models
for name, model in models.items():
    print(f"Training {name}...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
    model.save(f'models/{name}_mnist.h5')
    print(f"{name} saved to models/{name}_mnist.h5")