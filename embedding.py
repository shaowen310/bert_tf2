import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """

    def __init__(
        self, vocab_size, embedding_size=128, initializer_range=0.02, **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range

        ## init variables
        self.embedding_table = tf.Variable(
            tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)(
                [self.vocab_size, self.embedding_size]
            )
        )

    def call(self, inputs):
        # [B, SEQ, EMB]
        output = tf.gather(self.embedding_table, inputs)

        return output

    def get_config(self):
        config = dict(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            initializer_range=self.initializer_range,
        )
        return dict(**super().get_config(), **config)

    def get_embedding_table(self):
        return self.embedding_table


class PositionEmbedding(tf.keras.layers.Layer):
    """Appends positional embeddings to a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """

    def __init__(
        self,
        max_position_embeddings=512,
        embedding_size=128,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_position_embeddings = max_position_embeddings
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range

        # init variables
        self.full_position_embeddings = tf.Variable(
            tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)(
                [self.max_position_embeddings, self.embedding_size]
            )
        )

    def call(self, inputs):
        if inputs.shape[-1] != self.embedding_size:
            raise ValueError("`inputs` width should be equal to `embedding_size`")

        seq_length = tf.shape(inputs)[-2]

        assert_op = tf.debugging.assert_less_equal(
            seq_length, self.max_position_embeddings
        )
        with tf.control_dependencies([assert_op]):
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(
                self.full_position_embeddings, [0, 0], [seq_length, -1]
            )
            num_dims = len(inputs.shape)

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([-1, self.embedding_size])
            position_embeddings = tf.reshape(
                position_embeddings, position_broadcast_shape
            )
            return inputs + position_embeddings
