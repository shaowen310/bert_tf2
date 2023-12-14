import tensorflow as tf


class NextSentencePrediction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
