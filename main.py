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

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
n_iter = 0
cv_accuracy = []

seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])

images_aug = seq.augment_images(X)
plt.imshow(images_aug[50])
plt.show()
'''
model = model.cnn_model(num_classes)
for train_idx, test_idx in skf.split(X, Y):
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]

    n_iter += 1
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train)
    history = model.fit(x_train, y_train, batch_size=batch_num, epochs=10)
    pred = model.predict(x_test)

    accuracy = np.around(accuracy_score(y_test, np.argmax(pred, axis = 1)), 4)
    print('\n#{} 정확도: {}, 학습데이터 크기: {}, 검증데이터 크기: {}'.format(n_iter, accuracy, x_train.shape[0], x_test.shape[0]))
    cv_accuracy.append(accuracy)

print('\n정확도 :', cv_accuracy)
print('평균 정확도:', np.mean(cv_accuracy))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)
'''