from modules import load
from modules import model
from sklearn.model_selection import train_test_split
from tensorflow import keras

num_classes = 10

X, Y = load.dataload('./img')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

model = model.cnn_model(num_classes)

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)