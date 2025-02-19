import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet():
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_alexnet():
    model = models.Sequential([
        # Layer 1: Conv + ReLU + MaxPool
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),  # Output: 14x14x32

        # Layer 2: Conv + ReLU + MaxPool
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),  # Output: 7x7x64

        # Layer 3: Conv + ReLU
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),  # Output: 7x7x128

        # Layer 4: Conv + ReLU
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),  # Output: 7x7x128

        # Layer 5: Conv + ReLU + MaxPool
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),  # Output: 3x3x256

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_vggnet():
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def build_googlenet():
    input_tensor = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = models.Model(input_tensor, output_tensor)
    return model

def build_resnet():
    input_tensor = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = models.Model(input_tensor, output_tensor)
    return model