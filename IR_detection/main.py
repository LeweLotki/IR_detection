import os
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
collections.Callable = collections.abc.Callable
import tensorflow as tf
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.regularizers import l1
from data_loader import training_data, training_targets, testing_data, testing_tagrets
from data_prep import resolution

os.system('cls')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(
    seed=0
)

print(np.shape(training_data))
print(np.shape(training_targets))

## Shuffle data
idx = np.random.permutation(len(training_data))

## Creating dataset with labels
x_train = training_data[idx,:,:,:]
y_train = training_targets[:,idx].T

idx = np.random.permutation(len(testing_data))

x_test = testing_data
y_test = testing_tagrets.T

def cnn_model():

    filters = 12
    kernel = (4,4)
    l1_rate = 1e-3

    model=Sequential()
    model.add(Conv2D(filters,kernel, padding='same',input_shape=(1,resolution,resolution), 
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_rate)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters,kernel, padding='same',input_shape=(1,resolution,resolution), 
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_rate)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters,kernel, padding='same',input_shape=(1,resolution,resolution), 
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_rate)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = adam_v2.Adam(learning_rate=1e-3)
    model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

model = cnn_model() 

batch_size = 20
epochs = 20
try:
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
finally:
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

    y_pred = model.predict(x_test)
    y_class = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()



