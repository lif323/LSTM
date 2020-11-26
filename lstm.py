import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import activations
from tensorflow.python.util.tf_export import keras_export

class LSTM_CELL(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        # lstm 维度
        self.units = units
        super(LSTM_CELL, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.lstm_w = self.add_weight(shape=(input_dim, self.units * 4), name='kernel',
            initializer=initializers.get('glorot_uniform'))

        self.lstm_u = self.add_weight(shape=(self.units, self.units * 4),
                                                name='recurrent_kernel',
                                                initializer=initializers.get('orthogonal'))
        self.lstm_b = self.add_weight(
            shape=(self.units * 4), name='bias',
            initializer=initializers.get('zeros'))

        self.lstm_recurrent_activation = activations.get('hard_sigmoid')
        self.lstm_activation = activations.get('tanh')

    def call(self, inputs, states_tm1):
        h_tm1, c_tm1 = states_tm1
        w_i, w_f, w_c, w_o = tf.split(self.lstm_w, num_or_size_splits=4, axis=1)
        u_i, u_f, u_c, u_o = tf.split(self.lstm_u, num_or_size_splits=4, axis=1)
        b_i, b_f, b_c, b_o = tf.split(self.lstm_b, num_or_size_splits=4, axis=0)
        # w x
        wx_i = tf.matmul(inputs, w_i)
        wx_f = tf.matmul(inputs, w_f)
        wx_c = tf.matmul(inputs, w_c)
        wx_o = tf.matmul(inputs, w_o)
        # u h
        uh_i = tf.matmul(h_tm1, u_i)
        uh_f = tf.matmul(h_tm1, u_f)
        uh_c = tf.matmul(h_tm1, u_c)
        uh_o = tf.matmul(h_tm1, u_o)
        # w x + u * h + b
        i_t = tf.add(wx_i, tf.add(uh_i, b_i))
        f_t = tf.add(wx_i, tf.add(uh_f, b_f))
        c_t = tf.add(wx_i, tf.add(uh_c, b_c))
        o_t = tf.add(wx_i, tf.add(uh_o, b_o))

        i = self.lstm_recurrent_activation(i_t)
        f = self.lstm_recurrent_activation(f_t)
        c = f * c_tm1 + i * self.lstm_activation(c_t)
        o = self.lstm_recurrent_activation(o_t)
        # 计算 h
        h = o * self.lstm_activation(c)
        return h, (h, c)
        return h, (h, c)

class Rnn(tf.keras.layers.Layer):
    def __init__(self, units=128):
        super(Rnn, self).__init__()
        self.cell = LSTM_CELL(units)
        self.init_state = None
    def build(self, input_shape):
        shape = input_shape.as_list()
        n_batch = shape[0]
        init_h = tf.zeros(shape=[n_batch, self.cell.units])
        init_c = init_h
        self.init_state = (init_h, init_c)

    def call(self, inputs):
        # time step
        ts = inputs.shape.as_list()[1]
        h, c = self.init_state
        for i in range(ts):
            h, (h, c) = self.cell(inputs[:, i], (h, c))
        return h

if __name__ == "__main__":
    a = tf.random.normal(shape=(100, 10, 50))
    rnn = Rnn(128)
    h = rnn(a)
    print(h.shape)


