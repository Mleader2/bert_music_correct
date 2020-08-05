# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python3
"""Utilities for building a LaserTagger TF model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from bert import modeling
from bert import optimization
import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
from curLine_file import curLine

class LaserTaggerConfig(modeling.BertConfig):
  """Model configuration for LaserTagger."""

  def __init__(self,
               use_t2t_decoder=True,
               decoder_num_hidden_layers=1,
               decoder_hidden_size=768,
               decoder_num_attention_heads=4,
               decoder_filter_size=3072,
               use_full_attention=False,
               **kwargs):
    """Initializes an instance of LaserTagger configuration.

    This initializer expects both the BERT specific arguments and the
    Transformer decoder arguments listed below.

    Args:
      use_t2t_decoder: Whether to use the Transformer decoder (i.e.
        LaserTagger_AR). If False, the remaining args do not affect anything and
        can be set to default values.
      decoder_num_hidden_layers: Number of hidden decoder layers.
      decoder_hidden_size: Decoder hidden size.
      decoder_num_attention_heads: Number of decoder attention heads.
      decoder_filter_size: Decoder filter size.
      use_full_attention: Whether to use full encoder-decoder attention.
      **kwargs: The arguments that the modeling.BertConfig initializer expects.
    """
    super(LaserTaggerConfig, self).__init__(**kwargs)
    self.use_t2t_decoder = use_t2t_decoder
    self.decoder_num_hidden_layers = decoder_num_hidden_layers
    self.decoder_hidden_size = decoder_hidden_size
    self.decoder_num_attention_heads = decoder_num_attention_heads
    self.decoder_filter_size = decoder_filter_size
    self.use_full_attention = use_full_attention


class ModelFnBuilder(object):
  """Class for building `model_fn` closure for TPUEstimator."""

  def __init__(self, config, num_tags, num_slot_tags,
               init_checkpoint,
               learning_rate, num_train_steps,
               num_warmup_steps, use_tpu,
               use_one_hot_embeddings, max_seq_length, drop_keep_prob, entity_type_num, slot_ratio):
    """Initializes an instance of a LaserTagger model.

    Args:
      config: LaserTagger model configuration.
      num_tags: Number of different tags to be predicted.
      init_checkpoint: Path to a pretrained BERT checkpoint (optional).
      learning_rate: Learning rate.
      num_train_steps: Number of training steps.
      num_warmup_steps: Number of warmup steps.
      use_tpu: Whether to use TPU.
      use_one_hot_embeddings: Whether to use one-hot embeddings for word
        embeddings.
      max_seq_length: Maximum sequence length.
    """
    self._config = config
    self._num_tags = num_tags
    self.num_slot_tags = num_slot_tags
    self._init_checkpoint = init_checkpoint
    self._learning_rate = learning_rate
    self._num_train_steps = num_train_steps
    self._num_warmup_steps = num_warmup_steps
    self._use_tpu = use_tpu
    self._use_one_hot_embeddings = use_one_hot_embeddings
    self._max_seq_length = max_seq_length
    self.drop_keep_prob = drop_keep_prob
    self.slot_ratio = slot_ratio
    self.intent_ratio = 1.0-self.slot_ratio
    self.entity_type_num = entity_type_num
    self.lstm_hidden_size = 128

  def _create_model(self, mode, input_ids, input_mask, segment_ids, labels,
                    slot_labels, labels_mask, drop_keep_prob, entity_type_ids, sequence_lengths):
    """Creates a LaserTagger model."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = modeling.BertModel(
        config=self._config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=self._use_one_hot_embeddings)

    final_layer = model.get_sequence_output()
    # final_hidden = model.get_pooled_output()

    if is_training:
        # I.e., 0.1 dropout
        # final_hidden = tf.nn.dropout(final_hidden, keep_prob=drop_keep_prob)
        final_layer = tf.nn.dropout(final_layer, keep_prob=drop_keep_prob)

    # 结合实体信息
    batch_size, seq_length = modeling.get_shape_list(input_ids)

    self.entity_type_embedding = tf.get_variable(name="entity_type_embedding",
                                                 shape=(self.entity_type_num, self._config.hidden_size),
                                                 dtype=tf.float32, trainable=True,
                                                 initializer=tf.random_uniform_initializer(
                                                     -self._config.initializer_range*100,
                                                     self._config.initializer_range*100, seed=20))

    with tf.init_scope():
        impact_weight_init = tf.constant(1.0 / self.entity_type_num, dtype=tf.float32, shape=(1, self.entity_type_num))
    self.impact_weight = tf.Variable(
        impact_weight_init, dtype=tf.float32, name="impact_weight")  # 不同类型的影响权重
    impact_weight_matrix = tf.tile(self.impact_weight, multiples=[batch_size * seq_length, 1])

    entity_type_ids_matrix1 = tf.cast(tf.reshape(entity_type_ids, [batch_size * seq_length, self.entity_type_num]), dtype=tf.float32)
    entity_type_ids_matrix = tf.multiply(entity_type_ids_matrix1, impact_weight_matrix)
    entity_type_emb = tf.matmul(entity_type_ids_matrix, self.entity_type_embedding)
    final_layer = final_layer + tf.reshape(entity_type_emb, [batch_size, seq_length, self._config.hidden_size]) # TODO TODO    # 0.7071067811865476是二分之根号二
    # final_layer = tf.concat([final_layer, tf.reshape(entity_type_emb, [batch_size, seq_length,self._config.hidden_size])], axis=-1)

    if is_training:
        final_layer = tf.nn.dropout(final_layer, keep_prob=drop_keep_prob)

    (output_fw_seq, output_bw_seq), ((c_fw,h_fw),(c_bw,h_bw)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=LSTMCell(self.lstm_hidden_size),
        cell_bw=LSTMCell(self.lstm_hidden_size),
        inputs=final_layer,
        sequence_length=sequence_lengths,
        dtype=tf.float32)
    layer_matrix = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
    final_hidden = tf.concat([c_fw, c_bw], axis=-1)

    layer_matrix= tf.contrib.layers.layer_norm(
            inputs=layer_matrix, begin_norm_axis=-1, begin_params_axis=-1)


    intent_logits = tf.layers.dense(
          final_hidden,
          self._num_tags,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
          name="output_projection")
    slot_logits = tf.layers.dense(layer_matrix, self.num_slot_tags,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="slot_projection")


    with tf.variable_scope("loss"):
      loss = None
      per_example_intent_loss = None
      per_example_slot_loss = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        per_example_intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=intent_logits)
        slot_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=slot_labels, logits=slot_logits)
        per_example_slot_loss = tf.truediv(
            tf.reduce_sum(slot_loss, axis=1),
            tf.cast(tf.reduce_sum(labels_mask, axis=1), tf.float32))

        # from tensorflow.contrib.crf import crf_log_likelihood
        # from tensorflow.contrib.crf import viterbi_decode
        # batch_size = tf.shape(slot_logits)[0]
        # print(curLine(), batch_size, tf.constant([self._max_seq_length]))
        # length_batch = tf.tile(tf.constant([self._max_seq_length]), [batch_size])
        # print(curLine(), batch_size, "length_batch:", length_batch)
        # per_example_slot_loss, self.transition_params = crf_log_likelihood(inputs=slot_logits,
        #                 tag_indices=slot_labels,sequence_lengths=length_batch)
        # print(curLine(), "per_example_slot_loss:", per_example_slot_loss) # shape=(batch_size,)
        # print(curLine(), "self.transition_params:", self.transition_params) # shape=(9, 9)

        loss = tf.reduce_mean(self.intent_ratio*per_example_intent_loss + self.slot_ratio*per_example_slot_loss)
      pred_intent = tf.cast(tf.argmax(intent_logits, axis=-1), tf.int32)
      pred_slot = tf.cast(tf.argmax(slot_logits, axis=-1), tf.int32)
      return (loss, per_example_slot_loss, pred_intent, pred_slot,
              batch_size, entity_type_emb, impact_weight_matrix, entity_type_ids_matrix, final_layer, slot_logits)

  def build(self):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      tf.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s", name, features[name].shape)
      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      segment_ids = features["segment_ids"]
      labels = None
      slot_labels = None
      labels_mask = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        labels = features["labels"]
        slot_labels = features["slot_labels"]
        labels_mask = features["labels_mask"]
      (total_loss, per_example_loss, pred_intent, pred_slot,
       batch_size, entity_type_emb, impact_weight_matrix, entity_type_ids_matrix, final_layer, slot_logits) = self._create_model(
          mode, input_ids, input_mask, segment_ids, labels, slot_labels, labels_mask, self.drop_keep_prob,
          features["entity_type_ids"], sequence_lengths=features["sequence_lengths"])

      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      scaffold_fn = None
      if self._init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                        self._init_checkpoint)
        if self._use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      # for var in tvars:
      #   tf.logging.info("Initializing the model from: %s",
      #                   self._init_checkpoint)
      #   init_string = ""
      #   if var.name in initialized_variable_names:
      #     init_string = ", *INIT_FROM_CKPT*"
      #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
      #                   init_string)

      output_spec = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, self._learning_rate, self._num_train_steps,
            self._num_warmup_steps, self._use_tpu)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)

      elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(per_example_loss, labels, labels_mask, predictions):
          """Compute eval metrics."""
          accuracy = tf.cast(
              tf.reduce_all(  # tf.reduce_all  相当于＂逻辑ＡＮＤ＂操作，找到输出完全正确的才算正确
                  tf.logical_or(
                      tf.equal(labels, predictions),
                      ~tf.cast(labels_mask, tf.bool)),
                  axis=1), tf.float32)
          return {
              # This is equal to the Exact score if the final realization step
              # doesn't introduce errors.
              "sentence_level_acc": tf.metrics.mean(accuracy),
              "eval_loss": tf.metrics.mean(per_example_loss),
          }

        eval_metrics = (metric_fn,
                        [per_example_loss, labels, labels_mask, pred_intent])
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions={'pred_intent': pred_intent, 'pred_slot': pred_slot,
                        'batch_size':batch_size, 'entity_type_emb':entity_type_emb, 'impact_weight_matrix':impact_weight_matrix, 'entity_type_ids_matrix':entity_type_ids_matrix,
                        'final_layer':final_layer, 'slot_logits':slot_logits},
                                    # 'intent_logits':intent_logits, 'entity_type_ids_matrix':entity_type_ids_matrix},
            scaffold_fn=scaffold_fn)
      return output_spec

    return model_fn
