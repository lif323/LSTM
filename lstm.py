import tensorflow as tf
from tensorflow.python.keras import activations
import os

class LstmCell(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        super(LstmCell, self).__init__(**kwargs)
        # lstm 维度
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # trainable_variables
        self.lstm_w = self.add_weight(shape=(input_dim, self.units * 4), name='kernel', initializer='glorot_uniform')
        self.lstm_u = self.add_weight(shape=(self.units, self.units * 4), name='recurrent_kernel', initializer='orthogonal')
        self.lstm_b = self.add_weight(shape=(self.units * 4), name='bias', initializer='zeros')

        # activations
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
        i_t = wx_i + uh_i + b_i
        f_t = wx_f + uh_f + b_f
        c_t = wx_c + uh_c + b_c
        o_t = wx_o + uh_o + b_o
        
        i = self.lstm_recurrent_activation(i_t)
        f = self.lstm_recurrent_activation(f_t)
        c = f * c_tm1 + i * self.lstm_activation(c_t)
        o = self.lstm_recurrent_activation(o_t)
        h = o * self.lstm_activation(c)
        return h, (h, c)


class LSTM(tf.keras.layers.Layer):
    def __init__(self, units=128, return_sequences=False):
        super(LSTM, self).__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.cell = LstmCell(units)

    def get_init_state(self, inputs):
        batch_size = tf.shape(inputs)[0]
        init_h = tf.zeros(shape=[batch_size, self.units])
        init_c = init_h
        # get the initial state
        return (init_h, init_c)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        # set time step as the first dimension
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        state = self.get_init_state(inputs[0])
        hidden_seq = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[0])
        for time_step in tf.range(tf.shape(inputs)[0]):
            h, state = self.cell(inputs[time_step], state)
            hidden_seq = hidden_seq.write(time_step, h)

        hidden_seq = hidden_seq.stack()
        last_hidden = hidden_seq[-1]
        hidden_seq = tf.transpose(hidden_seq, perm=[1, 0, 2])
        ret = tf.case([(tf.constant(self.return_sequences), lambda: hidden_seq)], default=lambda: last_hidden)
        return ret

if __name__ == "__main__":
    a = tf.random.normal(shape=(100, 10, 50))
    lstm = LSTM(128, return_sequences=True)
    o = lstm(a)
    print(o.shape)


