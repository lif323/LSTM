import tensorflow as tf
from tensorflow import keras
import lstm

import numpy as np
import matplotlib.pyplot as plt
import os


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")
    def call(self, x):
        print(x.shape)
        os._exit(0)
        # [batch_size, height*width], [4, 784]
        x = self.flatten(x)
        x 
        # [batch_size, d1.output_size], [4, 128]
        x = self.d1(x)
        # [batch_size, d2.output_size], [4, 10]
        x = self.d2(x)
        return x

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
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(4)
    # 定义模型
    model = MyModel()
    epochs = 5
    for epoch in range(epochs):
        # train
        train_loss.reset_states()
        train_acc.reset_states()
        # images [batch_size, height, width] (4, 28, 28)
        # labels [batch_size]
        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                # pred [batch_size, n_class] (4, 10)
                pred = model(images)
                loss_val = loss(labels, pred)
                train_loss.update_state(loss_val)
                train_acc.update_state(labels, pred)
            grad = tape.gradient(loss_val, model.trainable_variables)
            opti.apply_gradients(zip(grad, model.trainable_variables))
        print("Epoch: {}, loss: {}, acc: {}".format(epoch, train_loss.result(), train_acc.result()), end=", ")

if __name__ == "__main__":
    train()
