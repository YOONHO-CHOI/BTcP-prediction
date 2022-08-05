#%% Import modules
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, LSTM, Flatten, Conv1D, GRU
from tensorflow.keras.layers import Bidirectional

#%% Models
def RNNs(input_shape, output_shape):
    inputs = Input(shape = input_shape, name = 'X')
    X = SimpleRNN(units=96, return_sequences=True)(inputs)
    X = SimpleRNN(units=48, activation='relu', return_sequences=True)(X)
    X = SimpleRNN(units=24, activation='relu', return_sequences=True)(X)
    X = Flatten()(X)
    outputs = Dense(units= output_shape, activation='sigmoid')(X)
    model = Model(inputs = inputs, outputs = outputs)
    return model

def GRUs(input_shape, output_shape):
    inputs = Input(shape = input_shape, name = 'X')
    X = GRU(units=48, activation='tanh', return_sequences=True)(inputs)
    X = GRU(units=24, activation='tanh', return_sequences=True)(X)
    X = GRU(units=12, activation='tanh', return_sequences=True)(X)
    X = Flatten()(X)
    outputs = Dense(units = output_shape, activation='sigmoid')(X)
    model = Model(inputs = inputs, outputs = outputs)
    return model

def LSTMs(input_shape, output_shape):
    inputs = Input(shape = input_shape, name = 'X')
    X = LSTM(units=48, activation='tanh', return_sequences=True)(inputs)
    X = LSTM(units=24, activation='tanh', return_sequences=True)(X)
    X = LSTM(units=12, activation='tanh', return_sequences=True)(X)
    X = Flatten()(X)
    outputs = Dense(units = output_shape, activation='sigmoid')(X)
    model = Model(inputs = inputs, outputs = outputs)
    return model

def BiLSTMs(input_shape, output_shape):
    inputs = Input(shape = input_shape, name = 'X')
    X = Bidirectional(LSTM(units=24, activation='tanh', return_sequences=True))(inputs)
    X = Bidirectional(LSTM(units=16, activation='tanh', return_sequences=True))(X)
    X = Bidirectional(LSTM(units=12, activation='tanh', return_sequences=True))(X)
    X = Flatten()(X)
    outputs = Dense(units = output_shape, activation='sigmoid')(X)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def CNN_LSTMs(input_shape, output_shape):
    inputs = Input(shape =input_shape, name = 'X')
    X = Conv1D(filters=64, kernel_size=1,strides=1, activation='relu', padding='same')(inputs)
    X2 = Conv1D(filters=32, kernel_size=3,strides=1, activation='relu', padding='same')(X)
    X3 = Conv1D(filters=32, kernel_size=5,strides=1, activation='relu', padding='same')(X)
    X4 = tf.keras.layers.concatenate([X2,X3])
    X5 = Conv1D(filters=1, kernel_size=1,strides=1, activation='relu', padding='same')(X4)
    f_X = LSTM(units=output_shape*2, activation='tanh', return_sequences=True)(inputs)
    f_X2= Dense(units = 1)(f_X)
    outputs_ = tf.math.add(f_X2,X5)
    outputs  = tf.math.sigmoid(outputs_)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def Transformers(input_shape, output_shape):
    inputs = Input(shape=input_shape, name='X')
    qx = tf.keras.layers.Conv1D(24, 8, 1, padding='same', activation='relu')(inputs)
    ## self-attention
    attention = tf.keras.layers.MultiHeadAttention(num_heads=24, key_dim=4)(qx, qx)
    x = tf.keras.layers.Add()([qx, attention])  # residual connection
    layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ## Feed Forward
    x = tf.keras.layers.Conv1D(24, 8, 1, padding='same')(layernorm1)
    x2 = tf.keras.layers.Conv1D(24, 8, 1, padding='same')(x)
    x3 = tf.keras.layers.Flatten()(x2)
    outputs = Dense(units=output_shape, activation='sigmoid')(x3)
    model = Model(inputs=inputs, outputs=outputs)
    return model
