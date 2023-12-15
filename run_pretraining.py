# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by shaowen310
# 1. Remove TPU support
# 2. Replace tf.compat.v1.flags with argparse
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import modeling
import optimization
import tensorflow as tf

import bert
from bert import BertModel
from adaptor import NextSentencePrediction

logger = tf.get_logger()


def argparser():
    ap = argparse.ArgumentParser(
        "Run masked LM/next sentence masked_lm pre-training for BERT."
    )

    ## Required parameters
    ap.add_argument(
        "--bert_config_file",
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.",
    )

    ap.add_argument(
        "--input_file",
        required=True,
        help="Input TF example files (can be a glob or comma separated).",
    )

    ap.add_argument(
        "--output_dir",
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    ## Other parameters
    ap.add_argument(
        "--init_checkpoint",
        default=None,
        help="Initial checkpoint (usually from a pre-trained BERT model).",
    )

    ap.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.",
    )

    ap.add_argument(
        "--max_predictions_per_seq",
        type=int,
        default=20,
        help="Maximum number of masked LM predictions per sequence. "
        "Must match data generation.",
    )

    ap.add_argument("--do_train", action="store_true", help="Whether to run training.")

    ap.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )

    ap.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Total batch size for training.",
    )

    ap.add_argument(
        "--eval_batch_size", type=int, default=8, help="Total batch size for eval."
    )

    ap.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for Adam.",
    )

    ap.add_argument(
        "--num_train_steps", type=int, default=100000, help="Number of training steps."
    )

    ap.add_argument(
        "--num_warmup_steps", type=int, default=10000, help="Number of warmup steps."
    )

    ap.add_argument(
        "--save_checkpoints_steps",
        type=int,
        default=1000,
        help="How often to save the model checkpoint.",
    )

    ap.add_argument(
        "--iterations_per_loop",
        type=int,
        default=1000,
        help="How many steps to make in each estimator call.",
    )

    ap.add_argument(
        "--max_eval_steps", type=int, default=100, help="Maximum number of eval steps."
    )

    return ap


def model_fn_builder(
    bert_config,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        model = BertModel(**bert_config.to_dict())

        model(
            input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            training=is_training,
        )

        (
            masked_lm_loss,
            masked_lm_example_loss,
            masked_lm_log_probs,
        ) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_word_embedding_table(),
            masked_lm_positions,
            masked_lm_ids,
            masked_lm_weights,
        )

        (
            next_sentence_loss,
            next_sentence_example_loss,
            next_sentence_log_probs,
        ) = get_next_sentence_output(
            bert_config, model.get_pooled_output(), next_sentence_labels
        )

        total_loss = masked_lm_loss + next_sentence_loss

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(
                        init_checkpoint, assignment_map
                    )
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logger.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                masked_lm_example_loss,
                masked_lm_log_probs,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_example_loss,
                next_sentence_log_probs,
                next_sentence_labels,
            ):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]]
                )
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32
                )
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights,
                )
                masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights
                )

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]]
                )
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis=-1, output_type=tf.int32
                )
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions
                )
                next_sentence_mean_loss = tf.compat.v1.metrics.mean(
                    values=next_sentence_example_loss
                )

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }

            eval_metrics = (
                metric_fn,
                [
                    masked_lm_example_loss,
                    masked_lm_log_probs,
                    masked_lm_ids,
                    masked_lm_weights,
                    next_sentence_example_loss,
                    next_sentence_log_probs,
                    next_sentence_labels,
                ],
            )
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn,
            )
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_masked_lm_output(
    bert_config, input_tensor, output_weights, positions, label_ids, label_weights
):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.compat.v1.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.compat.v1.variable_scope("transform"):
            input_tensor = tf.compat.v1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=bert.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range
                ),
            )
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.compat.v1.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.compat.v1.zeros_initializer(),
        )
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32
        )

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    next_sentence_logit = NextSentencePrediction(
        initializer_range=bert_config.initializer_range,
        name="cls/seq_relationship",
    )

    logits = next_sentence_logit(input_tensor)

    log_probs = tf.nn.log_softmax(logits, axis=-1)

    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1]
    )
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(
    input_files, max_seq_length, max_predictions_per_seq, is_training, num_cpu_threads=4
):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions": tf.io.FixedLenFeature(
                [max_predictions_per_seq], tf.int64
            ),
            "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights": tf.io.FixedLenFeature(
                [max_predictions_per_seq], tf.float32
            ),
            "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        # TODO support parallel reading
        d = tf.data.TFRecordDataset(input_files)
        d = d.repeat()
        if is_training:
            d = d.shuffle(buffer_size=256)

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True,
            )
        )
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t

    return example


def main(args):
    logger.setLevel(tf._logging.INFO)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.ModelConfig.from_json_file(args.bert_config_file)

    tf.io.gfile.makedirs(args.output_dir)

    input_files = []
    for input_pattern in args.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    logger.info("*** Input Files ***")
    for input_file in input_files:
        logger.info("  %s" % input_file)

    tpu_cluster_resolver = None

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=args.output_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=args.iterations_per_loop,
            num_shards=8,
            per_host_input_for_training=is_per_host,
        ),
    )

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=args.num_train_steps,
        num_warmup_steps=args.num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False,
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=args.max_seq_length,
            max_predictions_per_seq=args.max_predictions_per_seq,
            is_training=True,
        )
        estimator.train(input_fn=train_input_fn, max_steps=args.num_train_steps)

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=args.max_seq_length,
            max_predictions_per_seq=args.max_predictions_per_seq,
            is_training=False,
        )

        result = estimator.evaluate(input_fn=eval_input_fn, steps=args.max_eval_steps)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    ap = argparser()
    args = ap.parse_args()

    main(args)
