import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
from NN_utility.autoencoders import autoencoder


#  Define last layer activation function
#  =========================================================================================
class NormalizeActivation(Layer):
    def __init__(self, epsilon=1e-10, **kwargs):
        super(NormalizeActivation, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / (tf.reduce_sum(inputs, axis=-1, keepdims=True) + self.epsilon)

    def get_config(self):
        config = super(NormalizeActivation, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config
#  Import dataHead and dataset
#  ========================================================================================

f = h5py.File('FeNiCrMn_all.h5', 'r')
dataset = f['features'][()]
f.close()

x = dataset[:, :6]
y = dataset[:, 6:]

x = np.delete(x, 2, axis=1)

'''
encoder, decoder = autoencoder(x, 8)

x = encoder.predict(x)
'''
#  Delete empty columns
#  ===========================================
emptyIndex = []

for i in range(np.shape(y)[1]):
    if sum(y[:, i]) == 0:
        emptyIndex.append(i)

y = np.delete(y, emptyIndex, axis=1)


#  Neural network data process
#  ===================================
num_inputs = 5
num_outputs = 15
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#  Build Neural Networks
#  ===============================================================================================
inputs = Input(shape=(num_inputs,))

x = Dense(128, activation='relu')(inputs)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(15, activation='relu')(x)
outputs = NormalizeActivation()(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=32000)
