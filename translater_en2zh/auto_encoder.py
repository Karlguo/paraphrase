"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

from AutoEncoder import transformer
from AutoEncoder import text_inputter

def model():
    dtype = tf.float32
    return transformer.Transformer(
        source_inputter=text_inputter.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        target_inputter=text_inputter.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,
        name="autoencoder")

