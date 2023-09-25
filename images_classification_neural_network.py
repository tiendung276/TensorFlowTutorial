import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load data from : https://www.tensorflow.org/tutorials/keras/classification?hl=vi
data = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_img = train_img / 255.0
test_img = test_img / 255.0

"""
plt.imshow(train_img[7], cmap=plt.cm.binary)
plt.show()"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.Dense(units=10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_img, train_labels, epochs=6)

test_loss, test_acc = model.evaluate(test_img, test_labels)
print('\nTest accuracy:', test_acc)

predictions = model.predict(test_img)
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_img[i], cmap=plt.cm.binary)
    plt.xlabel("Prediction : "+class_names[test_labels[i]])
    plt.title("Actual : "+class_names[np.argmax(predictions[i])])
    plt.show()
