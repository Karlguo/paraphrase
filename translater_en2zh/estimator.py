"""Functions for Estimator API integration."""

import copy
import json
import six
import sys
sys.path.append("..")

import tensorflow as tf

from translater_en2zh import decoding_infer
from translater_en2zh import decoding
from translater_en2zh import constants
from translater_en2zh.config import load_model
from opennmt.utils import hooks
from opennmt.utils import parallel
from opennmt.utils import compat

def get_pretrained_variables_map(checkpoint_file_path, ignore_scope=''):
  reader = tf.train.NewCheckpointReader(tf.train.latest_checkpoint(checkpoint_file_path))
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0].replace(ignore_scope, '') in saved_shapes])
  #var_names = sorted(saved_shapes)
  restore_vars = []
  name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
  restore_map = {}
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name.replace(ignore_scope, '')]:
        restore_vars.append(curr_var)
        restore_map[saved_var_name.replace(ignore_scope, '')] = curr_var
  return restore_map

def make_serving_input_fn(model, metadata=None):
  """Returns the serving input function.

  Args:
    model: An initialized :class:`opennmt.models.model.Model` instance.
    metadata: Optional data configuration (to be removed). Some inputters
      currently require to peek into some data files to infer input sizes.

  Returns:
    A callable that returns a ``tf.estimator.export.ServingInputReceiver``.
  """

  def _fn():
    local_model = copy.deepcopy(model)
    # This is a hack for SequenceRecordInputter that currently infers the input
    # depth from the data files.
    # TODO: This function should not require the training data.
    if metadata is not None and "train_features_file" in metadata:
      _ = local_model.features_inputter.make_dataset(metadata["train_features_file"])
    return local_model.features_inputter.get_serving_input_receiver()

  return _fn

def make_input_fn(model,
                  mode,
                  batch_size,
                  features_file,
                  labels_file=None,
                  batch_type="examples",
                  batch_multiplier=1,
                  bucket_width=None,
                  maximum_features_length=None,
                  maximum_labels_length=None,
                  shuffle_buffer_size=None,
                  single_pass=False,
                  num_shards=1,
                  shard_index=0,
                  num_threads=None,
                  prefetch_buffer_size=None,
                  return_dataset=True,
                  need_ae=False):
  """Creates the input function.

  Args:
    model: An initialized :class:`opennmt.models.model.Model` instance.
    mode: A ``tf.estimator.ModeKeys`` mode.
    batch_size: The batch size to use.
    features_file: The file containing input features.
    labels_file: The file containing output labels.
    batch_type: The training batching stragety to use: can be "examples" or
      "tokens".
    batch_multiplier: The batch size multiplier to prepare splitting accross
       replicated graph parts.
    bucket_width: The width of the length buckets to select batch candidates
      from. ``None`` to not constrain batch formation.
    maximum_features_length: The maximum length or list of maximum lengths of
      the features sequence(s). ``None`` to not constrain the length.
    maximum_labels_length: The maximum length of the labels sequence.
      ``None`` to not constrain the length.
    shuffle_buffer_size: The number of elements from which to sample.
    single_pass: If ``True``, makes a single pass over the training data.
    num_shards: The number of data shards (usually the number of workers in a
      distributed setting).
    shard_index: The shard index this input pipeline should read from.
    num_threads: The number of elements processed in parallel.
    prefetch_buffer_size: The number of batches to prefetch asynchronously. If
      ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
      older versions.
    return_dataset: Make the input function return a ``tf.data.Dataset``
      directly or the next element.

  Returns:
    The input function.

  See Also:
    ``tf.estimator.Estimator``.
  """
  batch_size_multiple = 1
  if batch_type == "tokens" and model.dtype == tf.float16:
    batch_size_multiple = 8

  def _fn():
    local_model = copy.deepcopy(model)

    if mode == tf.estimator.ModeKeys.PREDICT:
      if need_ae:
        dataset = local_model.examples_inputter.make_inference_dataset_ae(
          features_file,
          batch_size,
          bucket_width=bucket_width,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)
      else:
        dataset = local_model.examples_inputter.make_inference_dataset(
          features_file,
          batch_size,
          bucket_width=bucket_width,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)

    elif mode == tf.estimator.ModeKeys.EVAL:
      dataset = local_model.examples_inputter.make_evaluation_dataset(
          features_file,
          labels_file,
          batch_size,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      dataset = local_model.examples_inputter.make_training_dataset(
          features_file,
          labels_file,
          batch_size,
          batch_type=batch_type,
          batch_multiplier=batch_multiplier,
          batch_size_multiple=batch_size_multiple,
          shuffle_buffer_size=shuffle_buffer_size,
          bucket_width=bucket_width,
          maximum_features_length=maximum_features_length,
          maximum_labels_length=maximum_labels_length,
          single_pass=single_pass,
          num_shards=num_shards,
          shard_index=shard_index,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)

    if return_dataset:
      return dataset
    else:
      iterator = dataset.make_initializable_iterator()
      # Add the initializer to a standard collection for it to be initialized.
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
      return iterator.get_next()

  return _fn

def make_model_fn(trans_model,
                  ae_model = None,
                  infering=False,
                  eval_prediction_hooks_fn=None,
                  num_devices=1,
                  devices=None,
                  hvd=None):
  """Creates the model function.

  Args:
    model: An initialized :class:`opennmt.models.model.Model` instance.
    eval_prediction_hooks_fn: A callable that takes the model predictions
      during evaluation and return an iterable of evaluation hooks (e.g. for
      saving predictions on disk, running external evaluators, etc.).
    num_devices: The number of devices used for training.
    devices: The list of devices used for training, if known.
    hvd: Optional Horovod object.

  See Also:
    ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
    arguments and the returned value.
  """
  dispatcher = parallel.GraphDispatcher(
      num_devices=num_devices,
      daisy_chain_variables=trans_model.daisy_chain_variables,
      devices=devices)

  def _fn(features, labels, params, mode, config):
    """model_fn implementation."""
    local_model_trans = copy.deepcopy(trans_model)
    local_model_ae    = copy.deepcopy(   ae_model)
    decode_two = True

    if mode == tf.estimator.ModeKeys.TRAIN:
      features_shards = dispatcher.shard(features)
      labels_shards = dispatcher.shard(labels)
      losses_shards = dispatcher(
          _loss_op, local_model_trans, features_shards, labels_shards, params, mode)
      loss = _extract_loss(losses_shards)
      train_op = local_model_trans.optimize_loss(loss, params=params, hvd=hvd)
      extra_variables = []
      if isinstance(train_op, tuple):
        train_op, extra_variables = train_op

      training_hooks = []
      if extra_variables:
        training_hooks.append(hooks.VariablesInitializerHook(extra_variables))
      if config is not None:
        local_model_trans.examples_inputter.visualize(config.model_dir)
        features_length = local_model_trans.features_inputter.get_length(features)
        labels_length = (
            local_model_trans.labels_inputter.get_length(labels)
            if not trans_model.unsupervised else None)
        num_words = {}
        if features_length is not None:
          num_words["source"] = tf.reduce_sum(features_length)
        if labels_length is not None:
          num_words["target"] = tf.reduce_sum(labels_length)
        training_hooks.append(hooks.LogWordsPerSecondHook(
            num_words,
            every_n_steps=config.save_summary_steps,
            output_dir=config.model_dir))
      return tf.estimator.EstimatorSpec(
          mode,
          loss=loss,
          train_op=train_op,
          training_hooks=training_hooks)

    elif mode == tf.estimator.ModeKeys.EVAL:
      logits, predictions = local_model_trans(features, labels, params, mode)
      loss = local_model_trans.compute_loss(logits, labels, training=False, params=params)
      loss = _extract_loss(loss)
      eval_metric_ops = local_model_trans.compute_metrics(predictions, labels)
      evaluation_hooks = []
      if predictions is not None and eval_prediction_hooks_fn is not None:
        evaluation_hooks.extend(eval_prediction_hooks_fn(predictions))
      return tf.estimator.EstimatorSpec(
          mode,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          evaluation_hooks=evaluation_hooks)

    elif mode == tf.estimator.ModeKeys.PREDICT and not infering:
      _, predictions = local_model_trans(features, labels, params, mode)
      if "index" in features:
        predictions["index"] = features["index"]

      export_outputs = {}
      export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
          tf.estimator.export.PredictOutput(predictions))

      return tf.estimator.EstimatorSpec(
          mode,
          predictions=predictions,
          export_outputs=export_outputs)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      print ("==========================================")
      #print (local_model.name)
      print (json.dumps(params, indent=2))
      #print (constants.END_OF_SENTENCE_ID)
      #print (constants.END_OF_SENTENCE_TOKEN)
      #ae_features = {
      #          "ids": features["ids_ae"],
      #          "length": features["length_ae"],
      #          "tokens": features["tokens_ae"]
      #        }
      #ae_metadata = {
      #          "source_words_vocabulary":"../vocab/vocab.zh",
      #          "target_words_vocabulary":"../vocab/vocab.zh"
      #        }
      #local_model = load_model("../model/AutoEncoder", "auto_encoder.py", model_name=None, serialize_model=True)
      #local_model.initialize(ae_metadata)
      #_, predictions = local_model(ae_features, labels, params, mode)
      #if "index" in features:
      #  predictions["index"] = features["index"]
      #export_outputs = {}
      #export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
      #      tf.estimator.export.PredictOutput(predictions))
      #return tf.estimator.EstimatorSpec(mode,predictions=predictions,export_outputs=export_outputs)

      #local_model = local_model_ae
      #local_model = local_model_trans

      beam_size = params.get("beam_width", 1)
      def get_info (local_model, features):
        with tf.variable_scope(local_model.name, initializer=local_model._initializer(params)):
          if not compat.reuse():
            print ("REUSING")
            local_model._build()
          inputs = local_model.features_inputter.make_inputs(features)
          length = local_model.features_inputter.get_length(features)

          #inputs = local_model.features_inputter.make_inputs({"ids":features["ids"]})
          #length = local_model.features_inputter.get_length({"length":features["length"]})
          #_, predictions = local_model._call(features, labels, params, mode)
          with tf.variable_scope("encoder"):
            encoder_outputs, encoder_state, encoder_sequence_length = local_model.encoder.encode(
                    inputs,
                    sequence_length = length,
                    mode = mode
                    )
          with tf.variable_scope("decoder"):
            batch_size = tf.shape(tf.contrib.framework.nest.flatten(encoder_outputs)[0])[0]
            initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_size)
            memory = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_size)
            memory_sequence_length = tf.contrib.seq2seq.tile_batch(encoder_sequence_length, multiplier=beam_size)

            output_layer = local_model.output_layer
            if output_layer == None:
              output_layer = tf.layers.Dense(local_model.labels_inputter.vocabulary_size, use_bias=True, dtype=local_model.labels_inputter.dtype)
              output_layer.build([None, local_model.decoder.output_size])
            embedding_fn = lambda ids: tf.nn.embedding_lookup(local_model.labels_inputter.embedding, ids)
            step_fn, initial_state = local_model.decoder.step_fn(
                  tf.estimator.ModeKeys.PREDICT,
                  batch_size * beam_size,
                  initial_state=initial_state,
                  memory=memory,
                  memory_sequence_length=memory_sequence_length,
                  dtype=local_model.labels_inputter.dtype
              )
            def symbols_to_logits_fn(ids, step, state):
              inputs = embedding_fn(ids)
              returned_values = step_fn(step, inputs, state, tf.estimator.ModeKeys.PREDICT)
              outputs, state, attention = returned_values
              logits = output_layer(outputs)
              return logits, state, attention
        return symbols_to_logits_fn, initial_state, batch_size
      
      trans_features = {"ids":features["ids"], "length":features["length"]}
      ae_features = {"ids":features["ids_ae"], "length":features["length_ae"]}
      trans_symbols_to_logits_fn, trans_initial_state, batch_size = get_info(local_model_trans, trans_features)
      ae_symbols_to_logits_fn, ae_initial_state, _ = get_info(local_model_ae, ae_features)

      start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
      decoding_strategy = decoding_infer.BeamSearch(beam_size,length_penalty = 0,coverage_penalty = 0)
      sampler = decoding.BestSampler()

      low_rate = 0.2
      ids_ori = features["ids_ori"]
      pass_spe_token = tf.where(ids_ori>2)
      appear_pos = tf.stack([pass_spe_token[:,0], tf.cast(tf.gather_nd(ids_ori, pass_spe_token), tf.int64)], axis=1)
      low_prob = tf.sparse_to_dense(appear_pos, [batch_size * beam_size, local_model_trans.labels_inputter.vocabulary_size], low_rate, 0, False)
      print ("===============================================================")
      print (low_prob)
      low_prob = None

      sampled_ids, sampled_length = decoding_infer.dynamic_decode(
                    trans_symbols_to_logits_fn,
                    ae_symbols_to_logits_fn,
                    local_model_trans.name,
                    local_model_ae.name,
                    start_tokens,
                    end_id=constants.END_OF_SENTENCE_ID,
                    initial_state_trans=trans_initial_state,
                    initial_state_ae=ae_initial_state,
                    decoding_strategy=decoding_strategy,
                    sampler=sampler,
                    maximum_iterations=150,
                    minimum_iterations=0,
                    attention_history=False,
                    attention_size=None,
                    low_prob=low_prob
                  )
      sampled_length = tf.minimum(sampled_length + 1, tf.shape(sampled_ids)[2])
      target_vocab_rev = local_model_trans.labels_inputter.vocabulary_lookup_reverse()
      target_tokens = target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
      
      predictions = {
            "tokens": target_tokens,
            "length": sampled_length,
        }
      num_hypotheses = params.get("num_hypotheses", 1)
      if num_hypotheses > 0:
        if num_hypotheses > beam_size:
          raise ValueError("n_best cannot be greater than beam_width")
        for key, value in six.iteritems(predictions):
          predictions[key] = value[:, :num_hypotheses]

      if ae_model != None:
        ae_init_checkpoint = params["ae_init_checkpoint"]
        ae_variables_map = get_pretrained_variables_map(ae_init_checkpoint)
        tf.contrib.framework.init_from_checkpoint(ae_init_checkpoint, ae_variables_map)
      
      trans_init_checkpoint = params["trans_init_checkpoint"]
      trans_variables_map = get_pretrained_variables_map(trans_init_checkpoint)
      tf.contrib.framework.init_from_checkpoint(trans_init_checkpoint, trans_variables_map)

      if "index" in features:
        predictions["index"] = features["index"]

      export_outputs = {}
      export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
          tf.estimator.export.PredictOutput(predictions))

      return tf.estimator.EstimatorSpec(
          mode,
          predictions=predictions,
          export_outputs=export_outputs)

    else:
      raise ValueError("Invalid mode")

  return _fn

def _loss_op(model, features, labels, params, mode):
  """Single callable to compute the loss."""
  training = mode == tf.estimator.ModeKeys.TRAIN
  logits, _ = model(features, labels, params, mode)
  return model.compute_loss(logits, labels, training=training, params=params)

def _normalize_loss(num, den=None):
  """Normalizes the loss."""
  if isinstance(num, list):  # Sharded mode.
    if den is not None:
      assert isinstance(den, list)
      return tf.add_n(num) / tf.add_n(den)
    else:
      return tf.reduce_mean(num)
  elif den is not None:
    return num / den
  else:
    return num

def _extract_loss(loss):
  """Extracts and summarizes the loss."""
  if not isinstance(loss, tuple):
    actual_loss = _normalize_loss(loss)
    tboard_loss = actual_loss
  else:
    actual_loss = _normalize_loss(loss[0], den=loss[1])
    tboard_loss = _normalize_loss(loss[0], den=loss[2]) if len(loss) > 2 else actual_loss
  tf.summary.scalar("loss", tboard_loss)
  return actual_loss
