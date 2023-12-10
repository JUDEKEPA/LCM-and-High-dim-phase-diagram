import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
#import matplotlib.pyplot as plt
import numpy as np


def autoencoder(x, output_dim):
    #  Set output dim
    #  ================================================================================================
    X_train, X_val, y_train, y_test = train_test_split(x, x, test_size=0.2, random_state=0)
    input_dim = np.shape(x)[1]  # assuming 5 elements in your alloy composition
    encoding_dim = output_dim


    #  Design encoder network
    #  ==========================================================================
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # Define decoding layer
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)  # sigmoid ensures values between 0 and 1

    # Compile the autoencoder model
    autoencoder = Model(input_layer, decoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    history = autoencoder.fit(X_train, X_train,  # Autoencoders use the input as the target
                              epochs=1000,
                              batch_size=32000,
                              shuffle=True,
                              validation_data=(X_val, X_val))

    encoder = Model(input_layer, encoded)

    decoder_layer1 = autoencoder.layers[-2]
    decoder_layer2 = autoencoder.layers[-1]

    decoder_input = Input(shape=(encoding_dim,))
    decoded_output = decoder_layer2(decoder_layer1(decoder_input))
    decoder = Model(decoder_input, decoded_output)

    return encoder, decoder

