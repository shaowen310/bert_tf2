# %% [markdown]
# Import libraries
# %%
import tensorflow as tf

# %%[markdown]
# Decode function for TFRecordDataset
# %%
max_seq_length = 128
max_predictions_per_seq = 20

schema = {
    "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
    "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
}


def decode_fn(record_bytes):
    return tf.io.parse_single_example(record_bytes, schema)


# %%[markdown]
# Decode one example from TFRecordDataset
# %%
input_file = "./tmp/tf_examples.tfrecord"

input_files = []
for input_pattern in input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

d = tf.data.TFRecordDataset(input_files)
mapped_d = d.map(decode_fn)

record = next(iter(mapped_d))
print(record.keys())
# %%
