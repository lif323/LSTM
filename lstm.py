import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import activations
from tensorflow.python.util.tf_export import keras_export
import rnn

class LSTM_CELL(tf.keras.layers.Layer):
    def __init__(self, unites=256, **kwargs):
        # lstm 维度
        self.units = units
        self.state_size = unites
        super(LSTM_CELL, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4), name='kernel',
            initializer=initializers.get('glorot_uniform'))

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                                name='recurrent_kernel',
                                                initializer=initializers.get('orthogonal'))
        self.bias = self.add_weight(
            shape=(self.units * 4), name='bias',
            initializer=initializers.get('zeros'))

        self.recurrent_activation = activations.get('hard_sigmoid')
        self.activation = activations.get('tanh')


    def call(self, inputs, states):
        last_h = states[0]
        last_c = states[1]
        os._exit(0)
        k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
        b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
        # w x
        x_i = K.dot(inputs, k_i)
        x_f = K.dot(inputs, k_f)
        x_c = K.dot(inputs, k_c)
        x_o = K.dot(inputs, k_o)
        # w x + b
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)

        k_i, k_f, k_c, k_o = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        # w x + u * h + x
        i = self.recurrent_activation(x_i + K.dot(last_h, k_i))
        f = self.recurrent_activation(x_f + K.dot(last_h, k_f))
        c = f * last_c + self.activation(x_c + K.dot(last_h, k_c))
        o = self.recurrent_activation(x_o + K.dot(last_h, k_o))

        # 计算 h
        h = o * self.activation(c)
        return h, [h, c]

if __name__ == "__main__":
    units = 128
    lstm = LSTM_CELL(units)
    init_h = tf.zeros(shape=(100, units))
    init_c = init_h
    a = tf.random.normal(shape=(100, 10, 50))
    rnn = rnn.RNN(lstm)
    y = rnn(a, [init_h, init_c])
    print(y.shape)


