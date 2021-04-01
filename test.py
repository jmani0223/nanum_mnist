from modules import load
from modules import model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import cross_val_score
num_classes = 10
batch_num = 32

X, Y = load.dataload('./img')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=222)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model = model.cnn_model(num_classes)
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print(model.summary())

hist = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_num, epochs=15)
acc=[i * 100 for i in hist.history['accuracy']]
val_acc=[i * 100 for i in hist.history['val_accuracy']]
# summarize history for accuracy
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
plt.show()


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)