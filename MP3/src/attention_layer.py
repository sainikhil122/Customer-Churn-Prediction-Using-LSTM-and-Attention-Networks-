import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="HiX")
class HiXAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1)

    def call(self, inputs):

        scores = self.score_dense(inputs)

        weights = tf.nn.softmax(scores, axis=1)

        context = tf.reduce_sum(inputs * weights, axis=1)

        return context, weights

    def get_config(self):
        return super().get_config()