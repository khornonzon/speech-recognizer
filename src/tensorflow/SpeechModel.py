from tensorflow import keras
from LibDataset import LibriDatasetEng
import tensorflow as tf
from keras import Model
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.conv1 = keras.layers.Conv2D(32, 3, activation=tf.nn.relu)
        self.conv2 = keras.layers.Conv2D(32, 3, activation=tf.nn.relu)
        self.conv3 = keras.layers.Conv2D(32, 3, activation=tf.nn.relu)
        self.conv4 = keras.layers.Conv2D(32, 3, activation=tf.nn.relu)
    
        self.dense2 = keras.layers.Dense(128, activation=tf.nn.relu)

        self.dense3 = keras.layers.Dense(128,activation=tf.nn.relu)
        self.dense4 = keras.layers.Dense(30, activation=tf.nn.relu)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        sizes = x.size()
        x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = tf.nn.log_softmax(x, axis=2)
        return x
class CTCloss(tf.keras.losses.Loss):
    """ CTCLoss objec for training the model"""
    def __init__(self, name: str = 'CTCloss') -> None:
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """ Compute the training batch CTC loss value"""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss

model = CustomModel()
data = LibriDatasetEng()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=CTCloss(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

model.fit(data.specs, data.labels, batch_size=32)