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

"""Build BERT Examples from text (source, target) pairs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import re
import collections
import tensorflow as tf
import numpy as np
from bert import tokenization
from curLine_file import curLine




class BertExample(object):
  """Class for training and inference examples for BERT.

  Attributes:
    editing_task: The EditingTask from which this example was created. Needed
      when realizing labels predicted for this example.
    features: Feature dictionary.
  """

  def __init__(self, input_ids,
               input_mask,
               segment_ids,
               labels,
               slot_labels,
               labels_mask,
               entity_type_ids=None,
               entityTypeMap=None):
    input_len = len(input_ids)
    if not (input_len == len(input_mask) and input_len == len(segment_ids)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              input_len))
    self.entityTypeMap = entityTypeMap
    self.features = collections.OrderedDict([
        ('input_ids', input_ids),
        ('input_mask', input_mask),
        ('segment_ids', segment_ids),
        ('labels', labels),
        ('slot_labels', slot_labels),
        ('labels_mask', labels_mask),
        ('entity_type_ids', entity_type_ids),
        ('sequence_lengths', len(input_ids))
    ])
    # self._token_start_indices = token_start_indices

  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len = max_seq_length - len(self.features['input_ids'])
    for key in self.features:
      if key in ["labels", 'entity_type_ids', 'sequence_lengths']:
        continue
      pad_id = pad_token_id if key == 'input_ids' else 0
      self.features[key].extend([pad_id] * pad_len)
      if len(self.features[key]) != max_seq_length:
        raise ValueError('{} has length {} (should be {}).'.format(
            key, len(self.features[key]), max_seq_length))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""

    def int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    tf_features = collections.OrderedDict([
        (key, int_feature(val)) if type(val) is not int else (key, int_feature([val])) for key, val in self.features.items()
    ])
    return tf.train.Example(features=tf.train.Features(feature=tf_features))


def get_slot_tags(current_length, param, tokenizer):
  start_segment = re.findall("<[\w_]*>", param)
  end_segment = re.findall("</[\w_]*>", param)
  assert len(start_segment) == len(end_segment)
  slot_tags = ["O"] * current_length
  if len(start_segment) == 0:
    return slot_tags
  search_location = 0  # param的下标
  tag_location = 0  # query或slot_tags的下标
  for s, e in zip(start_segment, end_segment):
      entityType = s[1:-1]
      # print(curLine(), entityType, s, e)
      assert "</%s>" % entityType == e
      # print(curLine(), "search %s in %s" % (s, param[search_location:]), "search_location=", search_location)
      start_index = param[search_location:].index(s) + search_location # TODO
      # print(curLine(), "search %s in %s" % (s, param[search_location:]), "start_index=", start_index)
      end_index = param[search_location:].index(e)
      entity_info = param[start_index+ len(s):search_location+end_index]
      assert len(entity_info)>0, "entity_info:%s" % entity_info
      before,after = entity_info, entity_info
      if "||" in entity_info:
          before, after = entity_info.split("||")
      tag_location += len(tokenizer.tokenize(param[search_location:start_index])) # search_location:search_location+start_index]))
      # print(curLine(), "param[%d:%d]=%s" % (search_location, start_index, param[search_location:start_index]))
      # print(curLine(), param[search_location:start_index], tag_location, start_index) # search_location:search_location+start_index], tag_location, start_index)
      # print(curLine(), entity_info, "before:%s, after:%s" % (before, after))
      slot_tags[tag_location] = "B-%s" % entityType

      before_pieces_len = len(tokenizer.tokenize(before))
      search_location += end_index + len(e)

      if before_pieces_len > 1:
        for loc in range(tag_location+1, tag_location+before_pieces_len-1):
            slot_tags[loc] = "I-%s" % entityType
        slot_tags[tag_location+before_pieces_len-1] = "E-%s" % entityType
        # print(curLine(), tag_location+before_pieces_len-1, len(slot_tags), "slot_tags:", slot_tags)
      tag_location += before_pieces_len
  return slot_tags


class my_tokenizer_class(object):
    def __init__(self, vocab_file, do_lower_case):
        self.full_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)
    # 需要包装一下，因为如果直接对中文用full_tokenizer.tokenize，会忽略文本中的空格
    def tokenize(self, text):
        segments = text.split(" ")
        word_pieces = []
        for segId, segment in enumerate(segments):
            if segId > 0:
                word_pieces.append(" ")
            word_pieces.extend(self.full_tokenizer.tokenize(segment))
        return word_pieces
    def convert_tokens_to_ids(self, tokens):
        id_list = [self.full_tokenizer.vocab[t]
                   if t != " " else self.full_tokenizer.vocab["[unused20]"] for t in tokens]
        return id_list

class BertExampleBuilder(object):
  """Builder class for BertExample objects."""

  def __init__(self, label_map, vocab_file,
               max_seq_length, do_lower_case, slot_label_map, entity_type_list, get_entity_func=None):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags to tag IDs.
      vocab_file: Path to BERT vocabulary file.
      max_seq_length: Maximum sequence length.
      do_lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
    """
    self._label_map = label_map
    self.slot_label_map = slot_label_map
    self._tokenizer = my_tokenizer_class(vocab_file, do_lower_case=do_lower_case) # tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)

    self._max_seq_length = max_seq_length
    self._pad_id = self._get_pad_id()
    self.entity_type_list = entity_type_list
    self.entity_type_num = len(self.entity_type_list)
    self.get_entity_func = get_entity_func

  def build_bert_example(
      self,
      sources,
      target = None,
      location = None
  ):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.

    Returns:
      BertExample, or None if the conversion from text to tags was infeasible
      and use_arbitrary_target_ids_for_infeasible_examples == False.
    """
    # Compute target labels.
    sep_mark = '[SEP]'
    query = sources[-1]  # current query

    slot_tags = []
    source_tokens = self._tokenizer.tokenize(query)

    current_length = min(len(source_tokens), self._max_seq_length-2)
    labels_mask = [1] * current_length  # 有效的label为１，无效为０
    queryId2tokenId = []
    for tokenId, token in enumerate(source_tokens):
        token_length = len(token)
        if token.startswith("##", 0, 2):
            token_length -= 2
        elif "[UNK]" in token:
            token_length = 1
        for _ in range(token_length):
            queryId2tokenId.append(tokenId)
    if len(queryId2tokenId) != len(query):
        print(curLine(), len(query), "query:", query)
        print(curLine(), len(source_tokens), "source_tokens:", source_tokens)
        print(curLine(), len(queryId2tokenId), "queryId2tokenId:", queryId2tokenId)
        input(curLine())
    assert len(queryId2tokenId) == len(query)

    #  结合实体的词汇信息
    entityTypeMap = self.get_entity_func(query, self.entity_type_list)  # 实体树返回
    entity_type_ids = np.zeros([len(source_tokens), self.entity_type_num])
    for entity_type, entity_info_list in entityTypeMap.items():
        for entity_info in entity_info_list:
            entity_type_id = self.entity_type_list.index(entity_type)
            entity_before = entity_info["before"]
            assert entity_before in query
            entity_not_zhuanyi = entity_before.replace("*", "\*").replace("+", "\+")
            index_list = re.finditer(entity_not_zhuanyi, query)
            for index in index_list:
                # index.end()和index_end 都是这个实体结束后的下一个位置
                index_start = queryId2tokenId[index.start()]
                index_end = queryId2tokenId[index.end()-1]+1
                entity_type_ids[index_start:index_end, entity_type_id] = 1  # TODO  实体类型,未包含分词信息
    # print(curLine(), len(query), "query:", query)
    # print(curLine(), "entity_type_ids:\n", entity_type_ids)
    # input(curLine())
    if len(sources) > 1:  #  up context
      context_tokens = self._tokenizer.tokenize(sources[0]) + ['[SEP]']
      slot_tags.extend(["O"] * len(context_tokens))
      source_tokens = context_tokens + source_tokens
      labels_mask = [0] * len(context_tokens) + labels_mask
      entity_type_ids = np.concatenate([np.zeros([len(context_tokens), self.entity_type_num]),entity_type_ids], axis=0)


    # if len(tokens)>self._max_seq_length - 2:
    #   print(curLine(), "%d tokens is to long," % len(task.source_tokens), "truncate task.source_tokens:", task.source_tokens)
    #  截断到self._max_seq_length - 2
    tokens = self._truncate_list(source_tokens)
    input_tokens = ['[CLS]'] + tokens + ['[SEP]']

    labels_mask = self._truncate_list(labels_mask)
    assert sum(labels_mask) > 0
    labels_mask = [0] + labels_mask + [0]  # [1 if tt not in ['[CLS]', '[SEP]'] else 0 for tt in input_tokens ]

    entity_type_ids = entity_type_ids[-(self._max_seq_length-2):]
    entity_type_ids = np.concatenate([np.zeros([1,self.entity_type_num]),
             entity_type_ids, np.zeros([1,self.entity_type_num])], axis = 0)  # CLS and last SEP

    intent_label = None
    slot_labels = None
    slot_tags_info = ""
    if target is not None:
      intent, param = target
      if intent not in self._label_map:
        self._label_map[intent] = len(self._label_map)
      intent_label = self._label_map[intent]
      anntotate = get_slot_tags(current_length=current_length, param=param, tokenizer=self._tokenizer)
      slot_tags += anntotate
      assert len(source_tokens) == len(slot_tags), "len(tokens)=%d, len(slot_tags)=%d" % (len(tokens), len(slot_tags))

      slot_tags = ['O'] + self._truncate_list(slot_tags) + ['O']
      slot_labels = []
      for slot_tag in slot_tags:
        if slot_tag not in self.slot_label_map:
          self.slot_label_map[slot_tag] = len(self.slot_label_map)
        slot_labels.append(self.slot_label_map[slot_tag])
      assert len(input_tokens) == len(slot_labels), "len(input_tokens)=%d, len(slot_labels)=%d" % (len(input_tokens), len(slot_labels))

      slot_tags_str = " ".join(anntotate)
      source_tokens_str = " ".join(source_tokens)
      slot_tags_info = "%s\t%s" % (source_tokens_str, slot_tags_str)
    else:
      slot_labels = [0] * len(input_tokens)

    if sep_mark in tokens:
      context_len = 1 + tokens.index(sep_mark)
      segment_ids = [0] * (context_len+1) + [1] * (len(tokens) - context_len + 1)  # cls and sep
    else:
      segment_ids = [0] * len(input_tokens)
    assert len(segment_ids) == len(input_tokens) # TODO

    input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    if 20 in source_tokens:
        print(curLine(), len(input_tokens), "input_tokens:", input_tokens)
        print(curLine(), len(input_ids), "input_ids:", input_ids)
    input_mask = [1] * len(input_ids)


    entity_type_ids = np.concatenate([entity_type_ids, np.zeros([self._max_seq_length-len(input_ids),
            self.entity_type_num])], axis = 0)  # PAD
    entity_type_ids = np.reshape(entity_type_ids.astype(int), [-1]).tolist() # 要转化为一维度的int, long才能转化为ｔｅｎｓｏｒ

    example = BertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=intent_label,
        slot_labels=slot_labels,
        labels_mask=labels_mask,
        entity_type_ids=entity_type_ids,
        entityTypeMap=entityTypeMap)
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example, input_tokens, slot_tags_info

  def _split_to_wordpieces(self, tokens):
    """Splits tokens (and the labels accordingly) to WordPieces.

    Args:
      tokens: Tokens to be split.
      labels: Labels (one per token) to be split.

    Returns:
      3-tuple with the split tokens, split labels, and the indices of the
      WordPieces that start a token.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    # Index of each wordpiece that starts a new token.
    token_start_indices = []
    for i, token in enumerate(tokens):
      # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
      token_start_indices.append(len(bert_tokens) + 1)
      if token != "[SEP]":
        pieces = self._tokenizer.tokenize(token)
      else:
        pieces = ["[SEP]"]
      bert_tokens.extend(pieces)
    return bert_tokens, token_start_indices

  def _truncate_list(self, x):
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[-(self._max_seq_length - 2):]

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    except KeyError:
      return 0
