import tensorflow as tf

from embedding import Embedding, PositionEmbedding
from transformer import TransformerEncoder


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return lambda features: tf.nn.gelu(features, approximate=True)
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        embedding_size=768,
        type_vocab_size=16,
        max_position_embeddings=512,
        dropout=0.1,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_rate = dropout

        ## init layers
        self.word_embedding = Embedding(
            self.vocab_size,
            embedding_size=self.embedding_size,
            initializer_range=self.initializer_range,
            name="word_embedding",
        )

        self.token_type_embedding = Embedding(
            self.type_vocab_size,
            embedding_size=self.embedding_size,
            initializer_range=self.initializer_range,
            name="token_type_embeddings",
        )

        self.position_embedding = PositionEmbedding(
            max_position_embeddings=self.max_position_embeddings,
            embedding_size=self.embedding_size,
            initializer_range=self.initializer_range,
            name="position_embeddings",
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, token_type_ids=None):
        embedding_output = self.word_embedding(inputs)

        # Add positional embeddings and token type embeddings,
        # then layer normalize and perform dropout.

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=tf.shape(inputs), dtype=tf.int32)

        token_type_embeddings = self.token_type_embedding(token_type_ids)
        embedding_output += token_type_embeddings

        embedding_output = self.position_embedding(embedding_output)

        embedding_output = self.layer_norm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        return embedding_output

    def get_word_embedding_table(self):
        return self.word_embedding.get_embedding_table()


class BertModel(tf.keras.Model):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        type_vocab_size=16,
        max_position_embeddings=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        initializer_range=0.02,
        **kwargs
    ):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range

        ## init layers
        self.bert_embedding = BertEmbedding(
            self.vocab_size,
            embedding_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            dropout=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            name="embeddings",
        )

        self.transformer_encoder = TransformerEncoder(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            intermediate_act_fn=get_activation(self.hidden_act),
            hidden_dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_dropout_prob,
            initializer_range=self.initializer_range,
            name="encoder",
        )

        self.sequence_pool = tf.keras.layers.Dense(
            self.hidden_size,
            activation=tf.tanh,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="pooler",
        )

    def call(self, inputs, input_mask=None, token_type_ids=None):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
        embedding_output = self.bert_embedding(inputs, token_type_ids=token_type_ids)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        all_encoder_layers = self.transformer_encoder(
            embedding_output,
            input_mask=input_mask,
            do_return_all_layers=True,
        )

        sequence_output = all_encoder_layers[-1]

        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.sequence_pool(first_token_tensor)

        self.sequence_output = sequence_output
        self.pooled_output = pooled_output
        self.all_encoder_layers = all_encoder_layers

        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "all_encoder_layers": all_encoder_layers,
        }

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
            float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
            to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_word_embedding_table(self):
        return self.bert_embedding.get_word_embedding_table()

    def get_config(self):
        config = dict(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_dropout_prob=self.attention_dropout_prob,
            initializer_range=self.initializer_range,
        )
        return dict(**super().get_config(), **config)

    @classmethod
    def from_config(cls, config):
        cls.__init__(**config)
