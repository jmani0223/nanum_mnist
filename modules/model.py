from tensorflow import keras

def cnn_model(num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(10, 5, strides=(1, 1), padding='same', activation='relu',
                            input_shape=(256, 256, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(20, 5, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])
    '''
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size = batch_num, epochs=5)'''
    return model
