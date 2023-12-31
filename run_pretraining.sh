source conf.sh

BERT_BASE_DIR=$(pwd)

python run_pretrainingv2.py \
  --input_file=$BERT_BASE_DIR/tmp/tf_examples.tfrecord \
  --output_dir=$BERT_BASE_DIR/tmp/pretraining_output \
  --do_train \
  --do_eval \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/checkpoint/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
  