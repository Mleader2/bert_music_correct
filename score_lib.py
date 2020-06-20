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
"""Utility functions for computing evaluation metrics."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import re
from typing import List, Text, Tuple

import sari_hook
import utils

import tensorflow as tf
from curLine_file import normal_transformer, curLine

def read_data(
    path,
    lowercase,
    target_domain_name):
  """Reads data from prediction TSV file.

  The prediction file should contain 3 or more columns:
  1: sources (concatenated)
  2: prediction
  3-n: targets (1 or more)

  Args:
    path: Path to the prediction file.
    lowercase: Whether to lowercase the data (to compute case insensitive
      scores).

  Returns:
    Tuple (list of sources, list of predictions, list of target lists)
  """
  sources = []
  predDomain_list = []
  predIntent_list = []
  domain_list = []
  right_intent_num = 0
  right_slot_num = 0
  exact_num = 0
  with tf.gfile.GFile(path) as f:
    for lineId, line in enumerate(f):
      if "sessionId" in line and "pred" in line:
        continue
      sessionId, query, predDomain,predIntent,predSlot, domain,intent,Slot = line.rstrip('\n').split('\t')
      # if target_domain_name == predDomain and target_domain_name != domain: # domain 错了
      #   print(curLine(), lineId, predSlot, "Slot:", Slot, "predDomain:%s, domain:%s" % (predDomain, domain), "predIntent:",
      #         predIntent)
      if predIntent == intent:
        right_intent_num += 1
        if predSlot == Slot:
          exact_num += 1

      if predSlot == Slot:
        right_slot_num += 1
      else:
        if target_domain_name == predDomain and target_domain_name == domain:  #
          print(curLine(), predSlot, "Slot:", Slot, "predDomain:%s, domain:%s" % (predDomain, domain), "predIntent:", predIntent)
      predDomain_list.append(predDomain)
      predIntent_list.append(predIntent)
      domain_list.append(domain)
  return predDomain_list, predIntent_list, domain_list, right_intent_num, right_slot_num, exact_num


def compute_exact_score(predictions, domain_list):
  """Computes the Exact score (accuracy) of the predictions.

  Exact score is defined as the percentage of predictions that match at least
  one of the targets.

  Args:
    predictions: List of predictions.
    target_lists: List of targets (1 or more per prediction).

  Returns:
    Exact score between [0, 1].
  """

  correct_domain = 0
  for p, d in zip(predictions, domain_list):
      if p==d:
        correct_domain += 1
  return correct_domain / max(len(predictions), 0.1)  # Avoids 0/0.


def compute_sari_scores(
    sources,
    predictions,
    target_lists,
    ignore_wikisplit_separators = True,
    tokenizer=None):
  """Computes SARI scores.

  Wraps the t2t implementation of SARI computation.

  Args:
    sources: List of sources.
    predictions: List of predictions.
    target_lists: List of targets (1 or more per prediction).
    ignore_wikisplit_separators: Whether to ignore "<::::>" tokens, used as
      sentence separators in Wikisplit, when evaluating. For the numbers
      reported in the paper, we accidentally ignored those tokens. Ignoring them
      does not affect the Exact score (since there's usually always a period
      before the separator to indicate sentence break), but it decreases the
      SARI score (since the Addition score goes down as the model doesn't get
      points for correctly adding <::::> anymore).

  Returns:
    Tuple (SARI score, keep score, addition score, deletion score).
  """
  sari_sum = 0
  keep_sum = 0
  add_sum = 0
  del_sum = 0
  length_sum = 0
  length_max = 0
  for source, pred, targets in zip(sources, predictions, target_lists):
    if ignore_wikisplit_separators:
      source = re.sub(' <::::> ', ' ', source)
      pred = re.sub(' <::::> ', ' ', pred)
      targets = [re.sub(' <::::> ', ' ', t) for t in targets]
    source_ids = tokenizer.tokenize(source)  # utils.get_token_list(source)
    pred_ids = tokenizer.tokenize(pred)  # utils.get_token_list(pred)
    list_of_targets = [tokenizer.tokenize(t) for t in targets]
    length_sum += len(pred_ids)
    length_max = max(length_max, len(pred_ids))
    sari, keep, addition, deletion = sari_hook.get_sari_score(
        source_ids, pred_ids, list_of_targets, beta_for_deletion=1)
    sari_sum += sari
    keep_sum += keep
    add_sum += addition
    del_sum += deletion
  n = max(len(sources), 0.1)  # Avoids 0/0.
  return (sari_sum / n, keep_sum / n, add_sum / n, del_sum / n, length_sum/n, length_max)