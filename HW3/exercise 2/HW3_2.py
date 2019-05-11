import tensorflow as tf
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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

# Create the model
loss = "sparse_categorical_crossentropy"
activation = tf.nn.relu
regularization = tf.keras.regularizers.l1(0.001)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation = activation, kernel_regularizer = regularization))   # Hidden Layer 1
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(16, activation = activation, kernel_regularizer = regularization))   # Hidden Layer 2
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))   # Output Layer

model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])

# Train and evaluate the model
history = model.fit(features, label, epochs = 50, validation_split = 0.33)

# Plot training & validation accuracy values
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('ReLu (Cross Entropy) with L1')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.tight_layout()
plt.axis(xmin = 0, xmax = 50, ymin = 0, ymax = 1)

# Plot training & validation loss values
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.tight_layout()
plt.axis(xmin = 0, xmax = 50, ymin = 0, ymax = 2)
plt.show()

session.close()

