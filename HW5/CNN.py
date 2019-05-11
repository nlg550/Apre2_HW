from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img 
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

#========================= Preprocessing =========================#
train_dir = os.path.join(os.curdir, "data\\train")
test_dir = os.path.join(os.curdir, "data\\test")

# Scale all images and apply Image Augmentation
image_size = 128
batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0 / 255)
train_gen = datagen.flow_from_directory(train_dir, target_size = (image_size, image_size), batch_size = 32)
test_gen = datagen.flow_from_directory(test_dir, target_size = (image_size, image_size), batch_size = 32)

img_shape = (image_size, image_size, 3)

#========================= CNN =========================#
# Base Model (MobileNet V2)
base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape, include_top = False, weights = 'imagenet')
base_model.trainable = False   # Freeze the convolutional layers

# Creating our own model
model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(0.01)))
model.add(tf.keras.layers.Dense(512, activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(0.01)))
model.add(tf.keras.layers.Dense(4, activation = "softmax"))

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.0001), loss = "categorical_crossentropy", metrics = ['accuracy'])

#========================= Training =========================#
epochs = 3
 
steps_per_epoch = train_gen.n // batch_size
val_steps = test_gen.n // batch_size
 
history = model.fit(train_gen, steps_per_epoch = steps_per_epoch, workers = 4, epochs = epochs, validation_data = test_gen, validation_steps = val_steps)
 
acc = history.history['acc']
val_acc = history.history['val_acc']
 
loss = history.history['loss']
val_loss = history.history['val_loss']
 
plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
 
plt.subplot(2, 1, 2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

session.close()
