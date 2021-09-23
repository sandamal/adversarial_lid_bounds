'''
    Author: Sandamal on 12/5/21
    Description: Train a model on MNIST
'''

import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras import models
from keras import layers
from keras.utils import to_categorical  # this just converts the labels to one-hot class
from keras.callbacks import ModelCheckpoint

session = tf.Session()
keras.backend.set_session(session)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training Examples: %d" % len(x_train))
print("Test Examples: %d" % len(x_test))

n_classes = 10
inds = np.array([y_train == i for i in range(n_classes)])
f, ax = plt.subplots(2, 5, figsize=(10, 5))
ax = ax.flatten()
for i in range(n_classes):
    ax[i].imshow(x_train[np.argmax(inds[i])].reshape(28, 28))
    ax[i].set_title(str(i))
plt.show()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images_1d = x_train.reshape((60000, 28 * 28))
train_images_1d = train_images_1d.astype('float32') / 255

test_images_1d = x_test.reshape((10000, 28 * 28))
test_images_1d = test_images_1d.astype('float32') / 255

train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)

h = network.fit(train_images_1d,
                train_labels,
                epochs=20,
                batch_size=128,
                shuffle=True,
                callbacks=[ModelCheckpoint('MNIST.h5', save_best_only=True)])

score, acc = network.evaluate(test_images_1d,
                              test_labels,
                              batch_size=128)

print("Test Accuracy: %.5f" % acc)

network.save('MNIST.h5')

# summarize history for accuracy
plt.plot(h.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
