from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from sklearn.metrics import confusion_matrix, classification_report

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

output_filename = "ones_output"
output_filename = os.path.join(os.curdir, "plots\\latest\\" + output_filename)
f_output = open(output_filename, "w")


def get_num_classes():
    return len(os.listdir("data\\train"))


def get_target_names():
    return os.listdir("data\\train")


def generate_data_set():
    # ========================= Preprocessing =========================#
    train_dir = os.path.join(os.curdir, "data\\train")
    test_dir = os.path.join(os.curdir, "data\\test")

    # Scale all images and apply Image Augmentation
    image_size = 128
    batch_size = 64

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0 / 255, validation_split = 0.3)
    train_gen = datagen.flow_from_directory(train_dir, target_size = (image_size, image_size), batch_size = batch_size,
                                            shuffle = True)
    validation_gen = datagen.flow_from_directory(train_dir, target_size = (image_size, image_size), batch_size = batch_size,
                                                 shuffle = True, subset = 'validation')
    test_gen = datagen.flow_from_directory(test_dir, target_size = (image_size, image_size), batch_size = 1,
                                           shuffle = False)

    img_shape = (image_size, image_size, 3)

    return train_gen, validation_gen, test_gen, batch_size, img_shape


def create_cnn(img_shape, regularization = False, initialize = ''):
    # ========================= CNN =========================#
    # Base Model (MobileNet V2)
    base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape, include_top = False, weights = 'imagenet')
    base_model.trainable = False   # Freeze the base model (weights doesn't change)

    # Creating our own model
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    if not regularization:
        model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(512, activation = tf.nn.relu))
    else:
        initializer = None
        if initialize == 'ones':
            initializer = tf.keras.initializers.Ones()
        elif initialize == 'zeros':
            initializer = tf.keras.initializers.Zeros()
        elif initialize == 'xavier':
            initializer = tf.keras.initializers.glorot_uniform(seed = 3)

        model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu, kernel_initializer = initializer, kernel_regularizer = tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(rate = 0.3))
        model.add(tf.keras.layers.Dense(512, activation = tf.nn.relu, kernel_initializer = initializer, kernel_regularizer = tf.keras.regularizers.l1(0.001)))
        model.add(tf.keras.layers.Dropout(rate = 0.3))

    n_classes = get_num_classes()
    model.add(tf.keras.layers.Dense(n_classes, activation = "softmax"))
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
    return model


def train(model, train_gen, validation_gen, batch_size, n_epoch = 5):
    # ========================= Training =========================#
    steps_per_epoch = train_gen.n // batch_size
    val_steps = validation_gen.n // batch_size

    history = model.fit(train_gen, steps_per_epoch = steps_per_epoch, epochs = n_epoch, validation_data = validation_gen,
                        validation_steps = val_steps)
    return model, history


def plot_results(history, title):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    f_output.write("Training Acc:" + str(acc) + '\n')
    f_output.write("Training Loss:" + str(loss) + '\n')
    f_output.write("Validation Acc:" + str(val_acc) + '\n')
    f_output.write("Validation Loss:" + str(val_loss) + '\n')

    plt.figure(figsize = (8, 8))
    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.plot(acc, label = 'Training Accuracy')
    plt.plot(val_acc, label = 'Validation Accuracy')
    plt.legend(loc = 'lower right')
    plt.ylabel('Accuracy')
    plt.xlabel("Epochs")
    plt.ylim([min(plt.ylim()), 1.1])

    plt.subplot(2, 1, 2)
    plt.plot(loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.ylabel('Loss')
    plt.xlabel("Epochs")
    plt.ylim([0, max(plt.ylim())])
    title = os.path.join(os.curdir, "plots\\latest\\" + title)
    plt.savefig(title)
    plt.show()


def test(model, test_gen, batch_size):
    # ========================= Test =========================#
    test_steps = test_gen.n // batch_size
    test_loss, test_acc = model.evaluate(test_gen, steps = test_steps)
    print('TEST ACCURACY: ', test_acc)
    session.close()
    f_output.write("Test Acc:" + str(test_acc))
    f_output.write("Test Loss:" + str(test_loss))


def create_confusion_matrix(test_gen):
    filenames = test_gen.filenames
    nb_samples = len(filenames)

    Y_pred = model.predict_generator(test_gen, steps = nb_samples)
    y_pred = np.argmax(Y_pred, axis = 1)
    print('Confusion Matrix')
    confusion_m = confusion_matrix(test_gen.classes, y_pred)
    print(confusion_m)

    f_output.write("Confusion Matrix" + '\n')
    target_names = get_target_names()
    f_output.write("ENTRIES" + target_names.__repr__() + '\n')
    f_output.write(confusion_m.__repr__() + '\n')

    print('Classification Report')
    report = classification_report(test_gen.classes, y_pred, target_names = target_names)
    print(report)
    f_output.write("Classification Report" + '\n')
    f_output.write(report.__repr__() + '\n')


if __name__ == "__main__":
    train_gen, validation_gen, test_gen, batch_size, img_shape = generate_data_set()
    model = create_cnn(img_shape, regularization = True)
    model, history = train(model, train_gen, validation_gen, batch_size, n_epoch = 20)
    test(model, test_gen, batch_size)
    create_confusion_matrix(test_gen)
    plot_results(history, "Ones Initialization")

    f_output.close()
