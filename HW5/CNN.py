from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from sklearn.metrics import confusion_matrix, classification_report


def generate_data_set():
    # ========================= Preprocessing =========================#
    train_dir = os.path.join(os.curdir, "data\\train")
    test_dir = os.path.join(os.curdir, "data\\test")

    # Scale all images and apply Image Augmentation
    image_size = 128
    batch_size = 64

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0 / 255, validation_split = 0.2)
    train_gen = datagen.flow_from_directory(train_dir, target_size = (image_size, image_size), batch_size = batch_size,
                                            shuffle = True, subset = 'training')
    validation_gen = datagen.flow_from_directory(train_dir, target_size = (image_size, image_size), batch_size = batch_size,
                                                 shuffle = True, subset = 'validation')
    test_gen = datagen.flow_from_directory(test_dir, target_size = (image_size, image_size), batch_size = batch_size,
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
        
        if initialize == 'ones': initializer = tf.keras.initializers.Ones()
        elif initialize == 'zeros': initializer = tf.keras.initializers.Zeros()
        elif initialize == 'xavier': initializer = tf.keras.initializers.glorot_uniform(seed = 3)

        model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu, kernel_initializer = initializer,
                                        kernel_regularizer = tf.keras.regularizers.l1(0.001)))
        model.add(tf.keras.layers.Dropout(rate = 0.3))
        model.add(tf.keras.layers.Dense(512, activation = tf.nn.relu, kernel_initializer = initializer,
                                        kernel_regularizer = tf.keras.regularizers.l1(0.001)))
        model.add(tf.keras.layers.Dropout(rate = 0.3))

    n_classes = len(os.listdir("data\\train"))
    model.add(tf.keras.layers.Dense(n_classes, activation = "softmax"))
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
    return model


def train(model, train_gen, validation_gen, batch_size, n_epoch = 5):
    # ========================= Training =========================#
    steps_per_epoch = train_gen.n // batch_size
    val_steps = validation_gen.n // batch_size

    history = model.fit_generator(train_gen, steps_per_epoch = steps_per_epoch, epochs = n_epoch, validation_data = validation_gen,
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
    plt.ylim([min(plt.ylim()) - 0.1, 1.1])

    plt.subplot(2, 1, 2)
    plt.plot(loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.ylabel('Loss')
    plt.xlabel("Epochs")
    plt.ylim([0, max(plt.ylim())])
    title = os.path.join(os.curdir, "plots\\" + title)
    plt.savefig(title)
    # plt.show()


def test(model, test_gen, batch_size):
    test_steps = test_gen.n // batch_size
    test_loss, test_acc = model.evaluate_generator(test_gen, steps = test_steps)
    print('TEST ACCURACY: ', test_acc)
    f_output.write("Test Acc:" + str(test_acc) + '\n')
    f_output.write("Test Loss:" + str(test_loss) + '\n')


def create_confusion_matrix(model, test_gen):
    filenames = test_gen.filenames
    nb_samples = len(filenames)

    Y_pred = model.predict_generator(test_gen, steps = nb_samples)
    y_pred = np.argmax(Y_pred, axis = 1)
    print('Confusion Matrix')
    confusion_m = confusion_matrix(test_gen.classes, y_pred)
    print(confusion_m)

    f_output.write("Confusion Matrix" + '\n')
    target_names = os.listdir("data\\train")
    f_output.write("ENTRIES" + target_names.__repr__() + '\n')
    f_output.write(confusion_m.__repr__() + '\n')

    print('Classification Report')
    report = classification_report(test_gen.classes, y_pred, target_names = target_names)
    print(report)
    f_output.write("Classification Report" + '\n')
    f_output.write(report.__repr__() + '\n')


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    
    train_gen, validation_gen, test_gen, batch_size, img_shape = generate_data_set()
    
    output_filename = "ones_output"
    output_filename = os.path.join(os.curdir, "plots\\" + output_filename)
    f_output = open(output_filename, "w")
    model = create_cnn(img_shape, regularization = True, initialize = "ones")
    model, history = train(model, train_gen, validation_gen, batch_size, n_epoch = 30)
    test(model, test_gen, batch_size)
    create_confusion_matrix(model, test_gen)
    plot_results(history, "Ones Initialization")
    f_output.close()
  
    output_filename = "zeros_output"
    output_filename = os.path.join(os.curdir, "plots\\" + output_filename)
    f_output = open(output_filename, "w")
    model = create_cnn(img_shape, regularization = True, initialize = 'zeros')
    model, history = train(model, train_gen, validation_gen, batch_size, n_epoch = 30)
    test(model, test_gen, batch_size)
    create_confusion_matrix(model, test_gen)
    plot_results(history, "Zeros Initialization")
    f_output.close()
  
    output_filename = "xavier_output"
    output_filename = os.path.join(os.curdir, "plots\\" + output_filename)
    f_output = open(output_filename, "w")
    model = create_cnn(img_shape, regularization = True, initialize = 'xavier')
    model, history = train(model, train_gen, validation_gen, batch_size, n_epoch = 30)
    test(model, test_gen, batch_size)
    create_confusion_matrix(model, test_gen)
    plot_results(history, "Xavier Initialization")
    f_output.close()
  
    output_filename = "no_regularization_output"
    output_filename = os.path.join(os.curdir, "plots\\" + output_filename)
    f_output = open(output_filename, "w")
    model = create_cnn(img_shape, regularization = False, initialize = '')
    model, history = train(model, train_gen, validation_gen, batch_size, n_epoch = 30)
    test(model, test_gen, batch_size)
    create_confusion_matrix(model, test_gen)
    plot_results(history, "No Regularization")
    f_output.close()

    output_filename = "regularization_output"
    output_filename = os.path.join(os.curdir, "plots\\" + output_filename)
    f_output = open(output_filename, "w")
    model = create_cnn(img_shape, regularization = True, initialize = '')
    model, history = train(model, train_gen, validation_gen, batch_size, n_epoch = 30)
    test(model, test_gen, batch_size)
    create_confusion_matrix(model, test_gen)
    plot_results(history, "Regularization")
    f_output.close()
    
    session.close()
