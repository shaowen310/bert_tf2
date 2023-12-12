from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations


class MultiHeadAttention(layers.Layer):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Scalar dimensions referenced here:
    - B = batch size (number of sequences)
    - F = `from_tensor` sequence length
    - T = `to_tensor` sequence length
    - N = `num_attention_heads`
    - H = `size_per_head`

    Args:
        - from_tensor: float Tensor of shape [batch_size, from_seq_length, from_width].
        - to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        - attention_mask: (optional) int32 Tensor of shape [batch_size, from_seq_length, to_seq_length].
        The values should be 1 or 0. The attention scores will effectively be set to -infinity
        for any positions in the mask that are 0, and will be unchanged for positions that are 1.
        - num_attention_heads: int. Number of attention heads.
        - size_per_head: int. Size of each attention head.
        - query_act: (optional) Activation function for the query transform.
        - key_act: (optional) Activation function for the key transform.
        - value_act: (optional) Activation function for the value transform.
        - attention_probs_dropout_prob: (optional) float. Dropout probability of the attention probabilities.
        - initializer_range: float. Range of the weight initializer.
        - do_return_2d_tensor: bool.
        If True, the output will be of shape [batch_size * from_seq_length, num_attention_heads * size_per_head].
        If False, the output will be of shape [batch_size, from_seq_length, num_attention_heads * size_per_head].
        - batch_size: (Optional) int. If the input is 2D, this might be the batch size of the 3D version
        of the `from_tensor` and `to_tensor`.
        - from_seq_length: (Optional) If the input is 2D, this might be the seq length of the 3D version
        of the `from_tensor`.
        - to_seq_length: (Optional) If the input is 2D, this might be the seq length of the 3D version
        of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads * size_per_head].
        (If `do_return_2d_tensor` is true, this will be of shape [batch_size * from_seq_length, num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def __init__(
        self,
        num_attention_heads=1,
        size_per_head=512,
        initializer_range=0.02,
        query_activation=None,
        key_activation=None,
        value_activation=None,
        attention_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.initializer_range = initializer_range
        self.query_act = query_activation
        self.key_act = key_activation
        self.value_act = value_activation
        self.attention_dropout = attention_dropout

        self.negative_infinity = -10000.0

        ## initialize layers
        dense_units = self.num_attention_heads * self.size_per_head

        # `query_layer` = [?, ? -> N*H]
        self.query_layer = layers.Dense(
            dense_units,
            activation=self.query_act,
            kernel_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="query",
        )

        # `key_layer` = [?, ? -> N*H]
        self.key_layer = layers.Dense(
            dense_units,
            activation=self.key_act,
            kernel_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="key",
        )

        # `value_layer` = [?, ? -> N*H]
        self.value_layer = layers.Dense(
            dense_units,
            activation=self.value_act,
            kernel_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="value",
        )

        self.dropout_layer = layers.Dropout(self.attention_dropout)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and 2 == len(input_shape)

    @staticmethod
    def create_attention_mask(from_shape, input_mask):
        """Creates 3D attention.

        Args:
            from_shape (tf.TensorShape): [batch_size, from_seq_length, ...]
            input_mask (tf.Tensor(dypte=tf.int32)): [batch_size, seq_length]

        Returns:
            tf.Tensor(dypte=tf.int32): 3D attention
        """
        # [B, 1, T]
        mask = tf.expand_dims(input_mask, axis=1)
        # [B, F, 1]
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)
        return ones * mask

    def call(
        self,
        inputs: List[tf.Tensor],
        attention_mask: Optional[tf.Tensor] = None,
        training=None,
        do_return_2d_tensor=False,
        batch_size=None,
        from_seq_length=None,
        to_seq_length=None,
    ):
        """_summary_

        Args:
            - inputs ([tf.Tensor(dtype=tf.float32)]): two float Tensors of shape [batch_size, seq_length, num_attention_heads*size_per_head]
            - attention_mask (tf.Tensor(dypte=tf.int32), optional): int32 Tensor of shape [batch_size, from_seq_length, to_seq_length].
            The values should be 1 or 0. The attention scores will effectively be set to -infinity for any positions in the mask that are 0,
            and will be unchanged for positions that are 1.

        Returns:
            tf.Tensor(dypte=tf.float32): float Tensor of shape [batch_size, from_seq_length, num_attention_heads * size_per_head].
            (If `do_return_2d_tensor` is true, this will be of shape [batch_size * from_seq_length, num_attention_heads * size_per_head]).
        """

        # [B, F, FW]
        from_tensor = inputs[0]
        # [B, T, TW]
        to_tensor = inputs[1]

        from_shape = tf.shape(from_tensor)
        to_shape = tf.shape(to_tensor)

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if batch_size is None or from_seq_length is None or to_seq_length is None:
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified."
                )

        # [B, F, FW] -> [B, F, N*H]
        query = self.query_layer(from_tensor)
        key = self.key_layer(to_tensor)
        # [B, T, TW] -> [B, T, N*H]
        value = self.value_layer(to_tensor)

        # [B, SEQ, N*H] -> [B, N, SEQ, H]
        def transpose_for_scores(input_tensor, seq_length):
            output_shape = [
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.size_per_head,
            ]
            output_tensor = tf.reshape(
                input_tensor,
                output_shape,
            )
            return tf.transpose(output_tensor, perm=[0, 2, 1, 3])

        # [B, F, N*H] -> [B, N, F, H]
        query = transpose_for_scores(query, from_seq_length)
        # [B, T, N*H] -> [B, N, T, H]
        key = transpose_for_scores(key, to_seq_length)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores /= tf.sqrt(float(self.size_per_head))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=1)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * self.negative_infinity

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        attention_probs = self.dropout_layer(attention_probs, training=training)

        # [B, T, N*H] -> [B, N, T, H]
        value = tf.reshape(
            value,
            [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head],
        )
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # `context_layer` = [B, N, F, H] -> [B, F, N, H]
        context_layer = tf.matmul(attention_probs, value)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        # [B, F, N, H] -> [B, F, N*H]
        if do_return_2d_tensor:
            context_layer = tf.reshape(
                context_layer,
                [
                    batch_size * from_seq_length,
                    self.num_attention_heads * self.size_per_head,
                ],
            )
        else:
            context_layer = tf.reshape(
                context_layer,
                [
                    batch_size,
                    from_seq_length,
                    self.num_attention_heads * self.size_per_head,
                ],
            )

        return context_layer

    def get_config(self):
        config = {
            "num_attention_heads": self.num_attention_heads,
            "size_per_head": self.size_per_head,
            "weight_initializer_range": self.initializer_range,
            "query_activation": activations.serialize(self.query_act),
            "key_activation": activations.serialize(self.key_act),
            "value_activation": activations.serialize(self.value_act),
            "attention_dropout": self.attention_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items())) + list(config.items())
