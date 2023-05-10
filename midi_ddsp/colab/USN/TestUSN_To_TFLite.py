def print_dict(d):
    for k, v in d.items():
        print(f"\t'{k}': {v}")

def print_dict2(d):
    l = 0
    for detail in d:
        print()
        print(f'item #{l}:')
        l = l + 1

        i = 0
        for k, v in detail.items():
            if i == 0:
                print(f"\t'{k}': {v}")
            else:
                print(f"\t\t'{k}': {v}")


import os
import platform
import sys


from sparsenet.core import sparse
import tensorflow as tf

print(tf.__version__)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

example_input = tf.convert_to_tensor(next(iter(x_train)))
example_input = tf.expand_dims(example_input, axis=0)


import gc

tf.keras.backend.clear_session()
gc.collect()

nunits=250
dens=0.3
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # tf.keras.layers.Dense(nunits, activation="relu"),
  sparse(units=nunits, density=dens, activation="relu"),
  tf.keras.layers.Dense(10, activation='softmax')
])

lr=1e-3
optimizer = tf.keras.optimizers.Nadam(lr =lr)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ns=12000
model.fit(x_train[:ns], y_train[:ns], epochs=4, batch_size=64,
          validation_data=(x_test[:ns], y_test[:ns]))

model.summary()

model_output = model(example_input)

print(model_output)
