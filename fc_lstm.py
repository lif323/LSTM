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
        self.lstm_u = self.add_weight(shape=(self.units, self.units * 4), name='recurrent_kernel_u', initializer='orthogonal')
        self.lstm_v = self.add_weight(shape=(self.units, self.units * 3), name='recurrent_kernel_v', initializer='orthogonal')
        self.lstm_b = self.add_weight(shape=(self.units * 4), name='bias', initializer='zeros')

        # activations
        self.sigmoid = activations.get('hard_sigmoid')
        self.tanh = activations.get('tanh')

    def call(self, inputs, states_tm1):
        h_tm1, c_tm1 = states_tm1
        w_i, w_f, w_c, w_o = tf.split(self.lstm_w, num_or_size_splits=4, axis=1)
        u_i, u_f, u_c, u_o = tf.split(self.lstm_u, num_or_size_splits=4, axis=1)
        v_i, v_f, v_o = tf.split(self.lstm_v, num_or_size_splits=3, axis=1)
        b_i, b_f, b_c, b_o = tf.split(self.lstm_b, num_or_size_splits=4, axis=0)
        # w * x + u * h + b
        i_t = self.sigmoid(tf.matmul(inputs, w_i) + tf.matmul(h_tm1, u_i) + tf.matmul(c_tm1, v_i) + b_i)
        f_t = self.sigmoid(tf.matmul(inputs, w_f) + tf.matmul(h_tm1, u_f) + tf.matmul(c_tm1, v_f) + b_f)


        c_t = f_t * c_tm1 + i_t * self.tanh(tf.matmul(inputs, w_c) + tf.matmul(h_tm1, u_c) + b_c)
        o_t = self.sigmoid(tf.matmul(inputs, w_o) + tf.matmul(h_tm1, u_o) + tf.matmul(c_t, v_o) + b_o)
        h_t = o_t * self.tanh(c_t)
        return h_t, (h_t, c_t)


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


