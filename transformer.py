import tensorflow as tf

from attention import MultiHeadAttention


class TransformerSingleEncoder(tf.keras.layers.Layer):
    @staticmethod
    def create_input_mask_from_input(input_tensor):
        # [B, SEQ]
        return tf.ones(shape=tf.shape(input_tensor)[:2], dtype=tf.int32)

    @staticmethod
    def create_attention_mask_from_input_mask(from_tensor, to_mask):
        """Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """

        # [B, 1, T]
        to_mask = tf.cast(tf.expand_dims(to_mask, axis=1), tf.float32)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.

        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(shape=tf.shape(from_tensor)[:2], dtype=tf.float32)
        broadcast_ones = tf.expand_dims(broadcast_ones, axis=2)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=tf.keras.activations.gelu,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.hidden_size = hidden_size
        size_per_head = self.hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range

        # layers
        self.attention = MultiHeadAttention(
            num_attention_heads=num_attention_heads,
            size_per_head=size_per_head,
            initializer_range=initializer_range,
            attention_dropout=attention_dropout,
            name="self",
        )

        self.layer_norm_att_out = tf.keras.layers.LayerNormalization(axis=-1)

        self.dense_to_interim = tf.keras.layers.Dense(
            self.intermediate_size,
            activation=self.intermediate_act_fn,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="intermediate",
        )
        self.dense_to_hidden = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="output",
        )
        self.dropout_hidden = tf.keras.layers.Dropout(self.hidden_dropout)
        self.layer_norm_out = tf.keras.layers.LayerNormalization(axis=-1)

    def build(self, input_shape):
        input_width = input_shape[-1]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError(
                f"The width of the input tensor ({input_width}) != hidden size ({self.hidden_size})"
            )

    def call(self, inputs, input_mask=None, training=None, seq_length=None, **kwargs):
        # [B, SEQ, HID]
        layer_input = inputs

        if input_mask is None:
            input_mask = self.create_input_mask_from_input(layer_input)

        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = self.create_attention_mask_from_input_mask(inputs, input_mask)

        # [[B, SEQ, HID], [B, SEQ, HID]] -> [B, SEQ, HID]; HID = N*H
        attention_output = self.attention(
            [layer_input, layer_input],
            attention_mask=attention_mask,
            training=training,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            **kwargs,
        )

        # Add a residual with `layer_input` and layer norm.
        attention_output = self.layer_norm_att_out(layer_input + attention_output)

        # The activation is only applied to the "intermediate" hidden layer.
        intermediate_output = self.dense_to_interim(attention_output)

        # Down-project back to `hidden_size` then add the residual.
        layer_output = self.dense_to_hidden(intermediate_output)
        layer_output = self.dropout_hidden(layer_output, training=training)
        layer_output = self.layer_norm_out(attention_output + layer_output)

        return layer_output

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "intermediate_act_fn": tf.keras.activations.serialize(
                self.intermediate_act_fn
            ),
            "hidden_dropout": self.hidden_dropout,
            "weight_initializer_range": self.initializer_range,
        }
        base_config = super().get_config()
        return dict(list(base_config.items())) + list(config.items())


class TransformerEncoder(tf.keras.layers.Layer):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
        - input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        - attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
        - hidden_size: int. Hidden size of the Transformer.
        - num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        - num_attention_heads: int. Number of attention heads in the Transformer.
        - intermediate_size: int. The size of the "intermediate" (a.k.a., feed forward) layer.
        - intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
        - hidden_dropout_prob: float. Dropout probability for the hidden tf.keras.layers.
        - attention_probs_dropout_prob: float. Dropout probability of the attention probabilities.
        - initializer_range: float. Range of the initializer (stddev of truncated normal).
        - do_return_all_layers: Whether to also return all layers or just the final layer.
    """

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn=tf.keras.activations.gelu,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers

        ## Init layers
        self.encoders = list()

        for layer_i in range(self.num_hidden_layers):
            self.encoders.append(
                TransformerSingleEncoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    intermediate_act_fn=intermediate_act_fn,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    initializer_range=initializer_range,
                    name=f"layer_{layer_i}",
                )
            )

    def call(self, inputs, input_mask=None, do_return_all_layers=False):
        all_layer_outputs = list()

        prev_output = inputs

        for layer_i in range(self.num_hidden_layers):
            prev_output = self.encoders[layer_i](
                prev_output,
                input_mask=input_mask,
            )
            all_layer_outputs.append(prev_output)

        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = layer_output
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = prev_output
            return final_output

    def get_config(self):
        config = {"num_hidden_layers": self.num_hidden_layers}

        base_config = super().get_config()
        return dict(list(base_config.items())) + list(config.items())
