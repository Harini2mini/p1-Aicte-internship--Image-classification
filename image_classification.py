import tensorflow as tf
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt

tf.__version__

np.__version__

mlt.__version__

"""**Load** Dataset


"""

from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

"""# Show some sample images of dataset with corrsponding **labels**

---


"""

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:10]]))
print('Corresponding classes for the labels: ' + str([cifar10_classes[x[0]] for x in y_train[0:10]]))

fig, axarr = plt.subplots(1, 10)
fig.set_size_inches(20, 6)

for i in range(10):
    image = x_train[i]
    axarr[i].imshow(image)

plt .show()

x_train.shape, y_train.shape, x_test.shape, y_test.shape

"""## Preparing the Dataset

Normalize the input data
"""
# Normalize the input data
x_train = x_train / 255.0
x_test = x_test / 255.0

from tensorflow import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.datasets import cifar10 

# Define the model
ann = keras.Sequential()
ann.add(Flatten(input_shape=(32, 32, 3)))  # Input layer
ann.add(Dense(2048, activation='relu'))  # Hidden layer
ann.add(Dense(10, activation='softmax'))  # Output layer

# Print model summary
ann.summary()

# Compile the model
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = ann.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

"""With the below simple function we will be able to plot our training **history**"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()  # Add this line to display the plot

"""# CNN **Model**"""

from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout # Change conv2D to Conv2D

cnn = keras.Sequential()

cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))
cnn.add(MaxPooling2D((2, 2)))

cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
cnn.add(MaxPooling2D((2, 2)))

cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
cnn.add(MaxPooling2D((2, 2)))

cnn.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
cnn.add(MaxPooling2D((2, 2)))

cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))  # Hidden layer
cnn.add(Dropout(0.3))
cnn.add(Dense(10, activation='softmax'))  # Output layer

cnn.summary()

# Compile the model
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = cnn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

def plotLosses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper right')
    plt.show()

plotLosses(history)

def plotAccuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

plotAccuracy(history)

from keras.models import load_model
cnn.save('my_model111.h5')
# Load the model
model = tf.keras.models.load_model('my_model111.h5')
import numpy as np
# Add a batch dimension to the input
x_test_sample = np.expand_dims(x_test[20], axis=0)

# Now pass it to the model for prediction
model.predict(x_test_sample)
plt.imshow(x_test[20])
# Example: if you have class names like this
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # replace with your actual class names

# Get the prediction probabilities
predictions = model.predict(x_test_sample)

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(predictions)

# Get the corresponding class name
predicted_class_name = class_names[predicted_class_index]

print(f"The predicted class is: {predicted_class_name}")