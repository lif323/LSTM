import tensorflow as tf
import time 
import lstm
import sys

class OfficeLSTM(tf.keras.Model):
    def __init__(self, lstm_units):
        super(OfficeLSTM, self).__init__()
        self.lstm_units = lstm_units
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(self.lstm_units))
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")

        # optimizer
        self.opti = tf.keras.optimizers.Adam()
        # loss function
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        

        self.metrics_loss = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.metrics_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    def call(self, x):
        x = self.rnn(x)
        # [batch_size, d1.output_size], [4, 128]
        x = self.d1(x)
        # [batch_size, d2.output_size], [4, 10]
        x = self.d2(x)
        return x
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        with tf.GradientTape() as tape:
            pred = self(data)
            loss_val = self.loss_obj(label, pred)
        self.metrics_loss.update_state(label, pred)
        self.metrics_acc.update_state(label, pred)
        grad = tape.gradient(loss_val, self.trainable_variables)
        self.opti.apply_gradients(zip(grad, self.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        pred = self(data)
        self.metrics_loss.update_state(label, pred)
        self.metrics_acc.update_state(label, pred)

class CustomLSTM(tf.keras.Model):
    def __init__(self, lstm_units):
        super(CustomLSTM, self).__init__()
        self.lstm_units = lstm_units
        self.rnn = lstm.LSTM(units=lstm_units)
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")

        # optimizer
        self.opti = tf.keras.optimizers.Adam()
        # loss function
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        # metrics
        self.metrics_loss = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.metrics_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    def call(self, x):
        x = self.rnn(x)
        # [batch_size, d1.output_size], [4, 128]
        x = tf.reshape(x, shape=[-1, self.lstm_units])
        x = self.d1(x)
        # [batch_size, d2.output_size], [4, 10]
        x = self.d2(x)
        return x
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        with tf.GradientTape() as tape:
            pred = self(data)
            loss_val = self.loss_obj(label, pred)
        self.metrics_loss.update_state(label, pred)
        self.metrics_acc.update_state(label, pred)
        grad = tape.gradient(loss_val, self.trainable_variables)
        self.opti.apply_gradients(zip(grad, self.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        pred = self(data)
        self.metrics_loss.update_state(label, pred)
        self.metrics_acc.update_state(label, pred)


def load_dataset(batch_size=32):
    # load data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # num used
    train_images = train_images[:6000]
    train_labels = train_labels[:6000]
    test_images = test_images[:1000]
    test_labels = test_labels[:1000]
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    train_images = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size)
    train_ds = tf.data.Dataset.zip((train_images, train_labels))

    test_images = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels).batch(batch_size)
    test_ds = tf.data.Dataset.zip((test_images, test_labels))

    return train_ds, test_ds

@tf.function
def test(model, dataset):
    for data, label in dataset:
        model.metrics_step(data, label)
    tf.print("test loss: ", model.metrics_loss.result(), "acc: ", model.metrics_acc.result())
    model.metrics_loss.reset_states()
    model.metrics_acc.reset_states()

@tf.function
def train(model, train_dataset, test_dataset, epoch=10):
    data_size = train_dataset.cardinality()
    for i, (data, label) in train_dataset.repeat(epoch).enumerate():
        model.train_step(data, label)
        if tf.equal(tf.math.floormod(i, data_size), 0):
            tf.print("epoch: ", i / data_size, "train loss: ", model.metrics_loss.result(), "acc: ", model.metrics_acc.result(), end=", ")
            model.metrics_loss.reset_states()
            model.metrics_acc.reset_states()
            test(model, test_dataset)

if __name__ == "__main__":
    train_ds, test_ds = load_dataset()
    test_custom_model = (int(sys.argv[1])==1)
    if test_custom_model:
        print("test custom lstm")
        model = CustomLSTM(lstm_units=200)
    else:
        print("test offical lstm")
        model = OfficeLSTM(lstm_units=200)
    
    start = time.time()
    train(model, train_ds, test_ds)
    end = time.time()
    print("total time consumption: ", end-start)
