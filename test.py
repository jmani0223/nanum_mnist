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
import cv2
import requests
num_classes = 10
batch_num = 16

#load.rename('./img2')
#load.crop('./img2')

X, Y = load.dataload('./ttt')

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=222)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=222)

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

seq = iaa.Sequential([
        #iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        #iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ],random_order=False)
plt.imshow(x_test[0])
plt.show()
images_aug = seq.augment_images(x_train)

x_train=np.concatenate((x_train, images_aug), axis=0)
y_train=np.concatenate((y_train, y_train), axis=0)

model = model.cnn_model(num_classes)
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print(model.summary())

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_num, epochs=100)
acc=[i * 100 for i in hist.history['accuracy']]
val_acc=[i * 100 for i in hist.history['val_accuracy']]


model.save("./model/mnist_model.h5")

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