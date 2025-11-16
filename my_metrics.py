import tensorflow as tf
from keras.layers import Layer

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    loss = tf.reduce_mean((1 - y_true) * 0.5 * square_pred + y_true * 0.5 * margin_square)
    return loss

class EuclideanDistance(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        featsA, featsB = inputs
        sum_squared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1, keepdims=True)
        return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_squared, tf.keras.backend.epsilon()))

    def get_config(self):
        config = super().get_config()
        return config
