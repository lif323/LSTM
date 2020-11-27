import tensorflow as tf
from tensorflow import keras
import lstm
import numpy as np
import os
import time
import pickle

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # office lstm
        #self.rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(256))
        # my lstm
        self.rnn = lstm.Rnn(256)

        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")
    def call(self, x):
        x = self.rnn(x)
        # [batch_size, d1.output_size], [4, 128]
        x = self.d1(x)
        # [batch_size, d2.output_size], [4, 10]
        x = self.d2(x)
        return x


@tf.function
def train_step(model, loss, opti, images, labels, train_loss, train_acc):
    with tf.GradientTape() as tape:
        # pred [batch_size, n_class] (4, 10)
        pred = model(images)
        loss_val = loss(labels, pred)
    train_loss.update_state(loss_val)
    train_acc.update_state(labels, pred)
    grad = tape.gradient(loss_val, model.trainable_variables)
    opti.apply_gradients(zip(grad, model.trainable_variables))



def train():
    # 定义优化器
    opti = tf.keras.optimizers.Adam()
    # 定义损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # 用于记录损失值
    train_loss = tf.keras.metrics.Mean()
    # 记录正确率
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    # 加载数据
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), _ = fashion_mnist.load_data()
    train_images = train_images / 255.0
    num_used = 5000
    train_images = train_images[:num_used]
    train_labels = train_labels[:num_used]
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(4)
    # 定义模型
    model = MyModel()
    epochs = 5

    list_time_cost = list()
    list_acc = list()
    for epoch in range(epochs):
        # train
        train_loss.reset_states()
        train_acc.reset_states()
        # images [batch_size, height, width] (4, 28, 28)
        # labels [batch_size]
        start = time.time()
        for images, labels in train_ds:
            train_step(model, loss, opti, images, labels, train_loss, train_acc)
        ends = time.time()
        cost = ends - start
        list_time_cost.append(cost)
        list_acc.append(train_acc.result().numpy())
        print("Time: {} s, Epoch: {}, loss: {}, acc: {}".format(cost, epoch, train_loss.result(), train_acc.result()))
    with open("./data/my_lstm_acc.pkl", "wb") as fw:
        pickle.dump(list_acc, fw)
    with open("./data/my_lstm_time_cost.pkl", "wb") as fw:
        pickle.dump(list_time_cost, fw)

if __name__ == "__main__":
    train()
