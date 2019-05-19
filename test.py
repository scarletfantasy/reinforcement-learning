import tensorflow as tf
import numpy as np
from tensorflow import keras
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train=x_train[:,:,:,None]
x_test=x_test[:,:,:,None]
model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (2, 2), (1, 1), padding='same', input_shape=(28, 28,1),kernel_regularizer=keras.regularizers.l2(0.01)),
  tf.keras.layers.ReLU(),
  tf.keras.layers.MaxPool2D((2, 2)),
  tf.keras .layers.Flatten(),

  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
print(model.apply(x_test))


