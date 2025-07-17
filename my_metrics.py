import tensorflow as tf
# import keras.backend as K
from keras.layers import Layer

# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def f1_score(y_true, y_pred):
#     prec = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     f1 = 2 * (prec * rec) / (prec + rec + K.epsilon())
#     return f1

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

# class CosineDistance(Layer):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, inputs):
#         featsA, featsB = inputs
        
#         dot_product = tf.keras.backend.sum(featsA * featsB, axis=1, keepdims=True)
        
#         normA = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(featsA), axis=1, keepdims=True))
#         normB = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(featsB), axis=1, keepdims=True))
        
#         normA = tf.keras.backend.maximum(normA, tf.keras.backend.epsilon())
#         normB = tf.keras.backend.maximum(normB, tf.keras.backend.epsilon())
        
#         cosine_similarity = dot_product / (normA * normB)
        
#         cosine_distance = 1 - cosine_similarity
        
#         return cosine_distance

#     def get_config(self):
#         config = super().get_config()
#         return config
