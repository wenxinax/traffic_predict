from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import os
os.environ['KERAS_BACKEND']='tensorflow'

def get_lstm(units):

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model