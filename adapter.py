import tensorflow as tf


class NextSentencePrediction(tf.keras.layers.Layer):
    """Simple binary classification. Note that 0 is "next sentence" and 1 is
    "random sentence". This weight matrix is not used after pre-training.
    """

    def __init__(self, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)

        self.initializer_range = initializer_range

        self.next_sentence_logit = tf.keras.layers.Dense(
            2,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
        )

    def __call__(self, inputs):
        logits = self.next_sentence_logit(inputs)
        return logits
