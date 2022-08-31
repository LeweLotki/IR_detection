import os
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
collections.Callable = collections.abc.Callable
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from data_loader import training_data, training_targets, testing_data, testing_tagrets
from sklearn.metrics import confusion_matrix
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import seaborn as sns

os.system('cls')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train = training_data
y_train = training_targets.T
x_test = testing_data
y_test = testing_tagrets.T

def cnn_model():
    model=Sequential()
    model.add(Conv2D(32,3,3, padding='same',input_shape=(1,128,128), activation='relu'))
    model.add(Conv2D(32,3,3, padding='same',input_shape=(1,128,128), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(64,3,3, padding='same',input_shape=(1,128,128), activation='relu'))
    model.add(Conv2D(64,3,3, padding='same',input_shape=(1,128,128), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(128,3,3, padding='same',input_shape=(1,128,128), activation='relu'))
    model.add(Conv2D(128,3,3, padding='same',input_shape=(1,128,128), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

model = cnn_model() 

batch_size = 25
epochs = 100
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

#confusion_mtx = confusion_matrix(y_true, y_class)
#fig, ax = plt.subplots(figsize=(15,10))
#ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
#ax.set_xlabel('Prediciton Label')
#ax.set_ylabel('True Label')
#ax.set_title('Confusion Matrix')
#plt.show()



