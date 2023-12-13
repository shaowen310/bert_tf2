import tensorflow as tf

from embedding import Embedding, PositionEmbedding


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
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)

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
        embedding_output = self.drop(embedding_output)

        return embedding_output

    def get_embedding_table(self):
        return self.word_embedding.get_embedding_table()
