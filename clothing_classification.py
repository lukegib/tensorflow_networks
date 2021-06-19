# TensorFlow libs
import tensorflow as tf
import tensorflow_datasets as tfds

# Helpers libs
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Load in the fashion dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset = dataset['train']
test_dataset = dataset['test']

# Names (for plotting)
class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

# Total data in each dataset
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Training examples: {}".format(num_train_examples))
print("Test examples: {}".format(num_test_examples))

# Preprocessing (normalize pixel values (0-255) to value in range 0-1)
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# Normalize each value in the datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Cache datasets for quicker training 
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input, tranforms 2D to 1D array
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # hidden
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # output, probabilities for each class
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# Now evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset: ', test_accuracy)

