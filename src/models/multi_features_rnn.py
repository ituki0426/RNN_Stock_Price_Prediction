# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout,GRU,Input,concatenate

def build_rnn_model(units = 50):
  open_input = Input(shape=(50, 1), name="Open")
  rnn_open = SimpleRNN(units = 50, return_sequences = True)(open_input)
  high_input = Input(shape=(50, 1), name="High")
  rnn_high = SimpleRNN(units = 50, return_sequences = True)(high_input)
  low_input = Input(shape=(50, 1), name="Low")
  rnn_low = SimpleRNN(units = 50, return_sequences = True)(low_input)
  close_input = Input(shape=(50, 1), name="Close")
  rnn_close = SimpleRNN(units = 50, return_sequences = True)(close_input)
  x = concatenate([rnn_open, rnn_high, rnn_low, rnn_close])
  x = Dense(units = 1)(x)
  model = Model(
      inputs = [open_input,high_input,low_input,close_input],
      outputs = [x]
  )
  model.compile(optimizer='adam', loss='mse')
  return model

if __name__ == "__main__":
    model = build_rnn_model()
    model.summary()