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
"""Utility functions for running inference with a LaserTagger model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from collections import defaultdict
import numpy as np
from find_entity.acmation import re_phoneNum
from curLine_file import curLine
from predict_param import get_slot_info_str, get_slot_info_str_forMusic

tongyin_yuzhi = 0.7
cancel_keywords = ["取消", "关闭", "停止", "结束", "关掉", "不要打", "退出", "不需要", "暂停", "谢谢你的服务"]

class LaserTaggerPredictor(object):
  """Class for computing and realizing predictions with LaserTagger."""

  def __init__(self, tf_predictor,
               example_builder,
               label_map, slot_label_map,
               target_domain_name):
    """Initializes an instance of LaserTaggerPredictor.

    Args:
      tf_predictor: Loaded Tensorflow model.
      example_builder: BERT example builder.
      label_map: Mapping from tags to tag IDs.
    """
    self._predictor = tf_predictor
    self._example_builder = example_builder
    self.intent_id_2_tag = {
        tag_id: tag for tag, tag_id in label_map.items()
    }
    self.slot_id_2_tag = {
        tag_id: tag for tag, tag_id in slot_label_map.items()
    }
    self.target_domain_name = target_domain_name

  def predict_batch(self, sources_batch, location_batch=None, target_domain_name=None, predict_domain_batch=[], raw_query=None):  # 由predict改成
    """Returns realized prediction for given sources."""
    # Predict tag IDs.
    keys = ['input_ids', 'input_mask', 'segment_ids', 'entity_type_ids', 'sequence_lengths']
    input_info = defaultdict(list)
    example_list = []
    input_tokens_list = []
    location = None
    for id, sources in enumerate(sources_batch):
      if location_batch is not None:
        location = location_batch[id]  # 表示是否能修改
      example, input_tokens, _= self._example_builder.build_bert_example(sources, location=location)
      if example is None:
        raise ValueError("Example couldn't be built.")
      for key in keys:
        input_info[key].append(example.features[key])
      example_list.append(example)
      input_tokens_list.append(input_tokens)

    out = self._predictor(input_info)
    intent_list = []
    slot_list = []

    for index, intent in enumerate(out['pred_intent']):
      predicted_intent_ids = intent.tolist()
      predict_intent = self.intent_id_2_tag[predicted_intent_ids]
      query = sources_batch[index][-1]
      slot_info = query


      predict_domain = predict_domain_batch[index]
      if predict_domain != target_domain_name:
        intent_list.append(predict_intent)
        slot_list.append(slot_info)
        continue
      for word in cancel_keywords:
        if word in query:
          if target_domain_name == 'navigation':
            predict_intent = 'cancel_navigation'
          elif target_domain_name == 'music':
            predict_intent = 'pause'
          elif target_domain_name == 'phone_call':
            predict_intent = 'cancel'
          else:
            raise "wrong target_domain_name:%s" % target_domain_name
          break

      if target_domain_name == 'navigation':
        if predict_intent != 'navigation':
          slot_info = query
        else:
          slot_info = self.get_slot_info(out['pred_slot'][index], example_list[index], input_tokens_list[index], query)
        # if "cancel" not in predict_intent:
        #   slot_info = self.get_slot_info(out['pred_slot'][index], example_list[index], input_tokens_list[index], query)
        #   if ">" in slot_info and "</" in slot_info:
        #     predict_intent = 'navigation'
      elif target_domain_name == 'phone_call':
        if predict_intent != 'make_a_phone_call':
          slot_info = query
        else:
          slot_info = self.get_slot_info(out['pred_slot'][index], example_list[index], input_tokens_list[index], query)
      elif target_domain_name == 'music':
        # slot_info = self.get_slot_info(out['pred_slot'][index], example_list[index], input_tokens_list[index], query)
        if predict_intent != 'play':
          slot_info = query
        else:
          slot_info = self.get_slot_info(out['pred_slot'][index], example_list[index], input_tokens_list[index], query)
      intent_list.append(predict_intent)
      slot_list.append(slot_info)
    return intent_list, slot_list

  def get_slot_info(self, slot, example, input_tokens, query):
    # 为了处理UNK问题
    token_index_map = {} # tokenId 对应的　word piece在query中的开始位置
    if "[UNK]" in input_tokens:
      start_index = 1
      for index in range(2, len(input_tokens)):
        if "[SEP]" in input_tokens[-index]:
          start_index = len(input_tokens)-index + 1
          break
      previous_id = 0
      for tokenizer_id, t in enumerate(input_tokens[start_index:-1], start=start_index):
        if tokenizer_id > 0 and "[UNK]" in input_tokens[tokenizer_id-1]:
          if t in query[previous_id:]:
            previous_id = previous_id + query[previous_id:].index(t)
          else:  # 出现连续的ＵＮＫ情况，目前的做法是假设长度为１
            previous_id += 1
        token_index_map[tokenizer_id] = previous_id
        if "[UNK]" not in t:
          length_t = len(t)
          if t.startswith("##", 0, 2):
            length_t -= 2
          previous_id += length_t
    predicted_slot_ids = slot.tolist()
    labels_mask = example.features["labels_mask"]
    assert len(labels_mask) == len(predicted_slot_ids)
    slot_info = []
    current_entityType = None
    index = -1
    # print(curLine(), len(predicted_slot_ids), "predicted_slot_ids:", predicted_slot_ids)
    for tokens, mask, slot_id in zip(input_tokens, labels_mask, predicted_slot_ids):
      index += 1
      if mask > 0:
        if tokens.startswith("##"):
          tokens = tokens[2:]
        elif "[UNK]" in tokens:  # 处理ＵＮＫ的情况
          previoud_id = token_index_map[index] # 　unk对应ｗｏｒｄ开始的位置
          next_previoud_id = previoud_id + 1 # 　unk对应word结束的位置
          if index+1 in token_index_map:
            next_previoud_id = token_index_map[index+1]
          tokens = query[previoud_id:next_previoud_id]
          print(curLine(), "unk self.passage[%d,%d]=%s" % (previoud_id, next_previoud_id, tokens))
        predict_slot = self.slot_id_2_tag[slot_id]
        # print(curLine(), tokens, mask, predict_slot)
        #  用规则增强对数字的识别
        if current_entityType == "phone_num" and "phone_num" not in predict_slot: # 正在phone_num的区间内
          token_numbers = re_phoneNum.findall(tokens)
          if len(token_numbers) > 0:
            first_span = token_numbers[0]
            if tokens.index(first_span) == 0:  # 下一个wore piece仍然以数字开头，应该加到前面那个区间中
              slot_info.append(first_span)
              if len(first_span) < len(tokens):  # 如果除了数字，还有其他部分则当作Ｏ来追加
                slot_info.append("</phone_num>")
                slot_info.append(tokens[len(first_span):])
                current_entityType = None
              continue

        if predict_slot == "O":
          if current_entityType is not None:  # 已经进入本区间
            slot_info.append("</%s>" % current_entityType)  # 结束
            current_entityType = None
          slot_info.append(tokens)
          continue

        token_type = predict_slot[2:]  # 当前ｔｏｋｅｎ的类型
        if current_entityType is not None and token_type != current_entityType:  # TODO 上一个区间还没结束
          slot_info.append("</%s>" % current_entityType)  # 强行结束
          current_entityType = None
        if "B-" in predict_slot:
          if token_type != current_entityType:  # 上一个本区间已经结束，则添加开始符号
            slot_info.append("<%s>" % token_type)
          slot_info.append(tokens)
          current_entityType = token_type
        elif "E-" in predict_slot:
          if token_type == current_entityType:  # 已经进入本区间
            if token_type not in {"origin","destination","destination","phone_num","contact_name","singer"} \
                or tokens not in {"的"}: #  TODO 某些实体一般不会以这些字结尾
              slot_info.append(tokens)
              slot_info.append("</%s>" % token_type)  # 正常情况，先添加再结束
            else:
              slot_info.append("</%s>" % token_type)  # 先结束再添加
              slot_info.append(tokens)
          else: #  类型不符合，当作Ｏ处理
            slot_info.append(tokens)
          current_entityType = None
        else:  # I
          if current_entityType != token_type and index+1 < len(predicted_slot_ids): # 前面没有Ｂ，直接以Ｉ开头．这时根据下一个ｔａｇ决定
            next_predict_slot = self.slot_id_2_tag[predicted_slot_ids[index+1]]
            # print(curLine(), "next_predict_slot:%s, token_type=%s" % (next_predict_slot, token_type))
            if token_type in next_predict_slot:
              slot_info.append("<%s>" % token_type)
              current_entityType = token_type
          slot_info.append(tokens)
    if current_entityType is not None:  # 已经到末尾还没结束
      slot_info.append("</%s>" % current_entityType)  # 强行结束
    if self.target_domain_name == "music":
      slot_info_str = get_slot_info_str_forMusic(slot_info, raw_query=query, entityTypeMap=example.entityTypeMap)
    else:
      slot_info_str = get_slot_info_str(slot_info, raw_query=query, entityTypeMap=example.entityTypeMap)
    return slot_info_str