"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt
import sys

sys.path.append("..")
from set2seq import transformer
from set2seq import text_inputter

def model():
    dtype = tf.float32
    return transformer.Transformer(
        source_inputter=text_inputter.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=300,
            dtype=dtype),
        target_inputter=text_inputter.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=300,
            dtype=dtype),
        num_layers_encoder=2,
        num_layers_decoder=2,
        num_units=256,
        num_heads=4,
        ffn_inner_dim=1024,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,
        name="set2seq")

