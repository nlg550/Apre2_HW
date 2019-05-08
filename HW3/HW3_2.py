import tensorflow as tf
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from dask.dataframe.tests.test_rolling import idx

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)


def importDataset(filename):
    file = open(filename, "r")
    
    if file.mode == "r":
        raw_data = file.readlines()
        data_size = len(raw_data)
              
        label = np.empty((data_size, 1), dtype = "int")
        features = np.empty((data_size, 9), dtype = "int")
        
        for i in range(data_size):
            
            line = raw_data[i].split(",")
            
            # Class
            if line[0] == "no-recurrence-events": label[i] = 0
            elif line[0] == "recurrence-events": label[i] = 1
            
            # Age (considering only the lower value of the range)
            if line[1] == "?" : features[i][0] = -1
            else: features[i][0] = np.fromstring(line[1], sep = "-")[0]
            
            # Menopause
            if line[2] == "?" : features[i][1] = -1
            elif line[2] == "lt40" : features[i][1] = 0
            elif line[2] == "ge40" : features[i][1] = 1
            elif line[2] == "premeno" : features[i][1] = 2
            
            # Tumor Size (considering only the lower value of the range)
            if line[3] == "?" : features[i][2] = -1
            else: features[i][2] = np.fromstring(line[3], sep = "-")[0]
            
            # Inv Node (considering only the lower value of the range)
            if line[4] == "?" : features[i][3] = -1
            else: features[i][3] = np.fromstring(line[4], sep = "-")[0]
            
            # Node Caps
            if line[5] == "?" : features[i][4] = -1
            elif line[5] == "yes" : features[i][4] = 1
            elif line[5] == "no" : features[i][4] = 0

            # Deg Malig
            if line[6] == "?" : features[i][5] = -1
            else: features[i][5] = int(line[6])
            
            # Breast
            if line[7] == "?" : features[i][6] = -1
            elif line[7] == "right" : features[i][6] = 1
            elif line[7] == "left" : features[i][6] = 0
            
            # Breast Quad
            if line[8] == "?" : features[i][7] = -1
            elif line[8] == "right_low" : features[i][7] = 0
            elif line[8] == "left_low" : features[i][7] = 1
            elif line[8] == "right_up" : features[i][7] = 2
            elif line[8] == "left_up" : features[i][7] = 3
            elif line[8] == "central" : features[i][7] = 4
            
            # Node Caps
            if line[9] == "?\n" : features[i][8] = -1
            elif line[9] == "yes\n" : features[i][8] = 1
            elif line[9] == "no\n" : features[i][8] = 0
        
        return features, label

        
features, label = importDataset("./breast-cancer.data")

# Fill the missing values with the median
imputer = SimpleImputer(missing_values = -1, strategy = "median")
features = imputer.fit_transform(features)

# Shuffle the data
idx = np.random.permutation(label.size)
label = label[idx]
features = features[idx]

epochs = 15

# Create the model for Sigmoid
loss1 = "sparse_categorical_crossentropy"
activation1 = tf.nn.sigmoid

model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(16, activation = activation1))   # Hidden Layer 1
model1.add(tf.keras.layers.Dense(16, activation = activation1))   # Hidden Layer 2
model1.add(tf.keras.layers.Dense(16, activation = activation1))   # Hidden Layer 3
model1.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))   # Output Layer

model1.compile(optimizer = 'adam', loss = loss1, metrics = ['accuracy'])

# Train and evaluate the model
history1 = model1.fit(features, label, epochs = epochs, validation_split = 0.25, verbose = 0, shuffle = False)

# Create the model for ReLU
loss2 = "sparse_categorical_crossentropy"
activation2 = tf.nn.relu

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(16, activation = activation2))   # Hidden Layer 1
model2.add(tf.keras.layers.Dense(16, activation = activation2))   # Hidden Layer 2
model2.add(tf.keras.layers.Dense(16, activation = activation2))   # Hidden Layer 3
model2.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))   # Output Layer

model2.compile(optimizer = 'adam', loss = loss2, metrics = ['accuracy'])

# Train and evaluate the model
history2 = model2.fit(features, label, epochs = epochs, validation_split = 0.25, verbose = 0, shuffle = False)

# Create the model for Leaky ReLU
loss3 = "sparse_categorical_crossentropy"
activation3 = tf.nn.leaky_relu

model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.Dense(16, activation = activation3))   # Hidden Layer 1
model3.add(tf.keras.layers.Dense(16, activation = activation3))   # Hidden Layer 2
model3.add(tf.keras.layers.Dense(16, activation = activation3))   # Hidden Layer 3
model3.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))   # Output Layer

model3.compile(optimizer = 'adam', loss = loss3, metrics = ['accuracy'])

# Train and evaluate the model
history3 = model3.fit(features, label, epochs = epochs, validation_split = 0.25, verbose = 0, shuffle = False)

# Plot training accuracy values
plt.subplot(2, 1, 1)
plt.plot(history1.history['acc'])
plt.plot(history2.history['acc'])
plt.plot(history3.history['acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Training values')
plt.legend(['Sigmod', 'ReLU' , 'Leaky ReLU'])
plt.tight_layout()
plt.axis(xmin = 0, xmax = epochs, ymin = 0, ymax = 1)

# Plot training loss values
plt.subplot(2, 1, 2)
plt.plot(history1.history['loss'])
plt.plot(history2.history['loss'])
plt.plot(history3.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Sigmod', 'ReLU' , 'Leaky ReLU'])
plt.tight_layout()
# plt.axis(xmin = 0, xmax = epochs, ymin = 0, ymax = 2)
plt.show()

# Plot validation accuracy values
plt.subplot(2, 1, 1)
plt.plot(history1.history['val_acc'])
plt.plot(history2.history['val_acc'])
plt.plot(history3.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Validation values')
plt.legend(['Sigmod', 'ReLU' , 'Leaky ReLU'])
plt.tight_layout()
plt.axis(xmin = 0, xmax = epochs, ymin = 0, ymax = 1)

# Plot validation loss values
plt.subplot(2, 1, 2)
plt.plot(history1.history['val_loss'])
plt.plot(history2.history['val_loss'])
plt.plot(history3.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Sigmod', 'ReLU' , 'Leaky ReLU'])
plt.tight_layout()
# plt.axis(xmin = 0, xmax = epochs, ymin = 0, ymax = 2)
plt.show()

print('Sigmod values on last epoch')
print('Training Accuracy: ' + str(history1.history['acc'][-1]))
print('Training Loss: ' + str(history1.history['loss'][-1]))
print('Validaiton Accuracy: ' + str(history1.history['val_acc'][-1]))
print('Validation Loss: ' + str(history1.history['val_loss'][-1]))
print('')
print('ReLU values on last epoch')
print('Training Accuracy: ' + str(history2.history['acc'][-1]))
print('Training Loss: ' + str(history2.history['loss'][-1]))
print('Validaiton Accuracy: ' + str(history2.history['val_acc'][-1]))
print('Validation Loss: ' + str(history2.history['val_loss'][-1]))
print('')
print('Leaky ReLU values on last epoch')
print('Training Accuracy: ' + str(history3.history['acc'][-1]))
print('Training Loss: ' + str(history3.history['loss'][-1]))
print('Validaiton Accuracy: ' + str(history3.history['val_acc'][-1]))
print('Validation Loss: ' + str(history3.history['val_loss'][-1]))

session.close()
