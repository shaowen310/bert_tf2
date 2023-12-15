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

        with tf.compat.v1.variable_scope("cls/seq_relationship"):
            output_weights = tf.compat.v1.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range),
            )
            output_bias = tf.compat.v1.get_variable(
                "output_bias", shape=[2], initializer=tf.compat.v1.zeros_initializer()
            )

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            labels = tf.reshape(labels, [-1])
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, per_example_loss, log_probs)
