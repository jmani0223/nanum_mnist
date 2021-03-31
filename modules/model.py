from tensorflow import keras

def cnn_model(num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                            input_shape=(256, 256, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])
    return model
