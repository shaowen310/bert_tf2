# BERT for TensorFlow 2
Re-implementation of [google-research/bert](https://github.com/google-research/bert) using TensorFlow 2.

## Dependencies

```bash
## Linux native
# https://www.tensorflow.org/install/pip#linux
# According to https://www.tensorflow.org/install/source#gpu
# for tf 2.14, install CUDA 12.2 and its respective drivers and
# recommend using conda environment to setup cudatoolkit and cudnn
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9
conda install -c conda-forge tensorflow=2.14

## NVIDIA container
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# nvcr.io/nvidia/tensorflow:23.10-tf2-py3
docker build -t berttrain .
```

## Training and evaluation

```bash
# native
bash run_pretraining.sh
# docker
bash run_docker_pretraining.sh
```

## Keras modules

`attention.py`: MultiHeadAttention

`transformer.py`: TransformerEncoder

`bert.py`: BertModel

`embedding.py`: BertEmbedding, PositionEmbedding, WordEmbedding

## TFRecordDataset

TFRecordDataset is byte storage. We need to map the records by schema when using the dataset. Refer to `dataset_study_jupy.py` for details.

## Tasks completed

1. Convert function-style model and layer definitions to object-style ones.
2. Replace `tf.compat.v1.flags` with `argparse`.
3. Replace tf1 logger with tf2 logger.

## Tasks TODO

1. Allow model saving and loading.
2. Use `keras` API instead of `estimator` API.

## References

1. [TF2 Making new layers and models via subclassing](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing)
2. [TF2 Customizing model saving and serialization](https://www.tensorflow.org/guide/keras/customizing_saving_and_serialization)
3. [TFRecords for Humans](https://planspace.org/20170323-tfrecords_for_humans/)
4. [TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
5. [How to train a Keras model on TFRecord files](https://keras.io/examples/keras_recipes/tfrecord/)
