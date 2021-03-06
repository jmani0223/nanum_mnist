from modules import load
from modules import model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
num_classes = 10
batch_num = 32
X, Y = load.dataload('./img')
'''
train_X, x_test, train_Y, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=222)
x_train, x_test, train_Y, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=222)
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=222)
n_iter = 0
cv_accuracy = []

seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        #iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])

images_aug = seq.augment_images(X)

cv =0
hist=[]
model = model.cnn_model(num_classes)
for train_idx, val_idx in skf.split(train_X, train_Y):
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]

    n_iter += 1
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train)
    hist = model.fit(x_train, y_train, batch_size=batch_num, epochs=20)
    print(hist.history.keys())
    #model.save_weights(`bottleneck_fc_model.h5`)
    pred = model.predict(x_val)

    accuracy = np.around(accuracy_score(y_val, np.argmax(pred, axis = 1)), 4)
    print('\n#{} 정확도: {}, 학습데이터 크기: {}, 검증데이터 크기: {}'.format(n_iter, accuracy, x_train.shape[0], x_val.shape[0]))
    cv_accuracy.append(accuracy)
    cv += 1
print('\n정확도 :', cv_accuracy)
print('평균 정확도:', np.mean(cv_accuracy))

plt.plot(hist.history['accuracy'])
plt.show()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)
'''