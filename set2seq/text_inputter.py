"""Define word-based embedders."""

import abc
import collections
import os
import six
import yaml
import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import lookup_ops as lookup
try:
  from tensorflow.contrib.tensorboard.plugins import projector
except ModuleNotFoundError:
  from tensorboard.plugins import projector

from google.protobuf import text_format

from opennmt import tokenizers
from set2seq import constants
from opennmt.inputters.inputter import Inputter
from opennmt.utils import compat
from opennmt.utils.cell import build_cell, last_encoding_from_state
from opennmt.utils.misc import count_lines


def visualize_embeddings(log_dir, embedding_var, vocabulary_file, num_oov_buckets=0):
  """Registers an embedding variable for visualization in TensorBoard.

  This function registers :obj:`embedding_var` in the ``projector_config.pbtxt``
  file and generates metadata from :obj:`vocabulary_file` to attach a label
  to each word ID.

  Args:
    log_dir: The active log directory.
    embedding_var: The embedding variable to visualize.
    vocabulary_file: The associated vocabulary file.
    num_oov_buckets: The number of additional unknown tokens.
  """
  # Copy vocabulary file to log_dir.
  basename = "%s.txt" % embedding_var.op.name.replace("/", "_")
  destination = os.path.join(log_dir, basename)
  tf.gfile.Copy(vocabulary_file, destination, overwrite=True)

  # Append <unk> tokens.
  with tf.gfile.Open(destination, mode="ab") as vocab:
    if num_oov_buckets == 1:
      vocab.write(b"<unk>\n")
    else:
      for i in range(num_oov_buckets):
        vocab.write(tf.compat.as_bytes("<unk%d>\n" % i))

  config = projector.ProjectorConfig()

  # If the projector file exists, load it.
  target = os.path.join(log_dir, "projector_config.pbtxt")
  if tf.gfile.Exists(target):
    with tf.gfile.Open(target, mode="rb") as target_file:
      text_format.Merge(target_file.read(), config)

  # If this embedding is already registered, just update the metadata path.
  exists = False
  for meta in config.embeddings:
    if meta.tensor_name == embedding_var.name:
      meta.metadata_path = basename
      exists = True
      break

  if not exists:
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = basename

  summary_writer = tf.summary.FileWriter(log_dir)

  projector.visualize_embeddings(summary_writer, config)

def load_pretrained_embeddings(embedding_file,
                               vocabulary_file,
                               num_oov_buckets=0,
                               with_header=True,
                               case_insensitive_embeddings=True):
  """Returns pretrained embeddings relative to the vocabulary.

  The :obj:`embedding_file` must have the following format:

  .. code-block:: text

      N M
      word1 val1 val2 ... valM
      word2 val1 val2 ... valM
      ...
      wordN val1 val2 ... valM

  or if :obj:`with_header` is ``False``:

  .. code-block:: text

      word1 val1 val2 ... valM
      word2 val1 val2 ... valM
      ...
      wordN val1 val2 ... valM

  This function will iterate on each embedding in :obj:`embedding_file` and
  assign the pretrained vector to the associated word in :obj:`vocabulary_file`
  if found. Otherwise, the embedding is ignored.

  If :obj:`case_insensitive_embeddings` is ``True``, word embeddings are assumed
  to be trained on lowercase data. In that case, word alignments are case
  insensitive meaning the pretrained word embedding for "the" will be assigned
  to "the", "The", "THE", or any other case variants included in
  :obj:`vocabulary_file`.

  Args:
    embedding_file: Path the embedding file. Entries will be matched against
      :obj:`vocabulary_file`.
    vocabulary_file: The vocabulary file containing one word per line.
    num_oov_buckets: The number of additional unknown tokens.
    with_header: ``True`` if the embedding file starts with a header line like
      in GloVe embedding files.
    case_insensitive_embeddings: ``True`` if embeddings are trained on lowercase
      data.

  Returns:
    A Numpy array of shape ``[vocabulary_size + num_oov_buckets, embedding_size]``.
  """
  # Map words to ids from the vocabulary.
  word_to_id = collections.defaultdict(list)
  with compat.gfile_open(vocabulary_file, mode="rb") as vocabulary:
    count = 0
    for word in vocabulary:
      word = word.strip()
      if case_insensitive_embeddings:
        word = word.lower()
      word_to_id[word].append(count)
      count += 1

  # Fill pretrained embedding matrix.
  with compat.gfile_open(embedding_file, mode="rb") as embedding:
    pretrained = None

    if with_header:
      next(embedding)

    for line in embedding:
      fields = line.strip().split()
      word = fields[0]

      if pretrained is None:
        pretrained = np.random.normal(
            size=(count + num_oov_buckets, len(fields) - 1))

      # Lookup word in the vocabulary.
      if word in word_to_id:
        ids = word_to_id[word]
        for index in ids:
          pretrained[index] = np.asarray(fields[1:])

  return pretrained

def _get_field(config, key, prefix=None, default=None, required=False):
  if prefix:
    key = "%s%s" % (prefix, key)
  value = config.get(key, default)
  if value is None and required:
    raise ValueError("Missing field '%s' in the data configuration" % key)
  return value


@six.add_metaclass(abc.ABCMeta)
class TextInputter(Inputter):
  """An abstract inputter that processes text."""

  def __init__(self,
               tokenizer=None,
               dtype=tf.float32,
               vocabulary_file_key=None,
               num_oov_buckets=0):
    super(TextInputter, self).__init__(dtype=dtype)
    self.tokenizer = tokenizer
    self.vocabulary_file_key = vocabulary_file_key
    self.num_oov_buckets = num_oov_buckets
    self.vocabulary = None
    self.vocabulary_size = None
    self.vocabulary_file = None

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    self.vocabulary_file = metadata[self.vocabulary_file_key]
    self.vocabulary_size = count_lines(self.vocabulary_file) + self.num_oov_buckets
    if self.tokenizer is None:
      tokenizer_config = _get_field(metadata, "tokenization", prefix=asset_prefix)
      if tokenizer_config:
        if isinstance(tokenizer_config, six.string_types) and compat.gfile_exists(tokenizer_config):
          with compat.gfile_open(tokenizer_config, mode="rb") as config_file:
            tokenizer_config = yaml.load(config_file, Loader=yaml.UnsafeLoader)
        self.tokenizer = tokenizers.OpenNMTTokenizer(params=tokenizer_config)
      else:
        self.tokenizer = tokenizers.SpaceTokenizer()
    self.tokenizer.initialize(metadata)
    return super(TextInputter, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)

  def export_assets(self, asset_dir, asset_prefix=""):
    return self.tokenizer.export_assets(asset_dir, asset_prefix=asset_prefix)

  def vocabulary_lookup(self):
    """Returns a lookup table mapping string to index."""
    return lookup.index_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        num_oov_buckets=self.num_oov_buckets,
        default_value=constants.UNKNOWN_ID)

  def vocabulary_lookup_reverse(self):
    """Returns a lookup table mapping index to string."""
    return lookup.index_to_string_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        default_value=constants.UNKNOWN_TOKEN)

  def make_dataset(self, data_file, training=None):
    self.vocabulary = self.vocabulary_lookup()
    return tf.data.TextLineDataset(data_file)

  def get_dataset_size(self, data_file):
    return count_lines(data_file)

  def make_features(self, element=None, features=None, training=None):
    """Tokenizes raw text."""
    if features is None:
      features = {}
    if "tokens" in features:
      return features
    tokens = self.tokenizer.tokenize(element)
    features["length"] = tf.shape(tokens)[0]
    features["tokens"] = tokens
    return features


class WordEmbedder(TextInputter):
  """Simple word embedder."""

  def __init__(self,
               vocabulary_file_key,
               embedding_size=None,
               embedding_file_key=None,
               embedding_file_with_header=True,
               case_insensitive_embeddings=True,
               trainable=True,
               dropout=0.0,
               tokenizer=None,
               dtype=tf.float32):
    """Initializes the parameters of the word embedder.

    Args:
      vocabulary_file_key: The data configuration key of the vocabulary file
        containing one word per line.
      embedding_size: The size of the resulting embedding.
        If ``None``, an embedding file must be provided.
      embedding_file_key: The data configuration key of the embedding file.
      embedding_file_with_header: ``True`` if the embedding file starts with a
        header line like in GloVe embedding files.
      case_insensitive_embeddings: ``True`` if embeddings are trained on
        lowercase data.
      trainable: If ``False``, do not optimize embeddings.
      dropout: The probability to drop units in the embedding.
      tokenizer: An optional :class:`opennmt.tokenizers.tokenizer.Tokenizer` to
        tokenize the input text. Defaults to a space tokenization.
      dtype: The embedding type.

    Raises:
      ValueError: if neither :obj:`embedding_size` nor :obj:`embedding_file_key`
        are set.

    See Also:
      The :meth:`opennmt.inputters.text_inputter.load_pretrained_embeddings`
      function for details about the pretrained embedding format and behavior.
    """
    super(WordEmbedder, self).__init__(
        tokenizer=tokenizer,
        dtype=dtype,
        vocabulary_file_key=vocabulary_file_key)
    self.embedding = None
    self.embedding_size = embedding_size
    self.embedding_file = None
    self.embedding_file_key = embedding_file_key
    self.embedding_file_with_header = embedding_file_with_header
    self.case_insensitive_embeddings = case_insensitive_embeddings
    self.trainable = trainable
    self.dropout = dropout

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    assets = super(WordEmbedder, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)
    if self.embedding_file_key is not None:
      self.embedding_file = metadata[self.embedding_file_key]
    else:
      embedding = _get_field(metadata, "embedding", prefix=asset_prefix)
      if embedding is None and self.embedding_size is None:
        raise ValueError("embedding_size must be set")
      if embedding is not None:
        self.embedding_file = embedding["path"]
        self.trainable = embedding.get("trainable", True)
        self.embedding_file_with_header = embedding.get("with_header", True)
        self.case_insensitive_embeddings = embedding.get("case_insensitive", True)
    if self.embedding_file is None and self.embedding_size is None:
      raise ValueError("Must either provide embedding_size or embedding_file_key")

    return assets

  def get_receiver_tensors(self):
    return {
        "tokens": tf.placeholder(tf.string, shape=(None, None)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

  def make_features(self, element=None, features=None, training=None):
    """Converts words tokens to ids."""
    features = super(WordEmbedder, self).make_features(
        element=element, features=features, training=training)
    if "ids" in features:
      return features
    if self.vocabulary is None:
      self.vocabulary = self.vocabulary_lookup()
    ids = self.vocabulary.lookup(features["tokens"])
    if not self.is_target:
      features["ids"] = ids
    else:
      bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=ids.dtype)
      eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=ids.dtype)
      features["ids"] = tf.concat([bos, ids], axis=0)
      features["ids_out"] = tf.concat([ids, eos], axis=0)
      features["length"] += 1 # Increment length accordingly.
    return features

  def build(self, input_shape=None):
    if self.embedding_file:
      pretrained = load_pretrained_embeddings(
          self.embedding_file,
          self.vocabulary_file,
          num_oov_buckets=self.num_oov_buckets,
          with_header=self.embedding_file_with_header,
          case_insensitive_embeddings=self.case_insensitive_embeddings)
      self.embedding_size = pretrained.shape[-1]
      initializer = tf.constant_initializer(value=pretrained.astype(self.dtype))
    else:
      initializer = tf.keras.initializers.glorot_uniform()
    shape = [self.vocabulary_size, self.embedding_size]
    self.embedding = tf.Variable(
        initial_value=lambda: initializer(shape, dtype=self.dtype),
        trainable=self.trainable,
        name=compat.name_from_variable_scope("w_embs"))
    super(WordEmbedder, self).build(input_shape)

  def make_inputs(self, features, training=None):
    if not self.built:
      self.build()
    outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
    if training and self.dropout > 0:
      outputs = tf.keras.layers.Dropout(self.dropout)(outputs, training=training)
    return outputs

  def visualize(self, log_dir):
    visualize_embeddings(
        log_dir,
        self.embedding,
        self.vocabulary_file,
        num_oov_buckets=self.num_oov_buckets)

