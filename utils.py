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
"""Utility functions for LaserTagger."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import csv

import tensorflow as tf
from curLine_file import curLine, normal_transformer, other_tag



def yield_sources_and_targets(input_file, input_format, domain_name):
  """Reads and yields source lists and targets from the input file.

  Args:
    input_file: Path to the input file.
    input_format: Format of the input file.

  Yields:
    Tuple with (list of source texts, target text).
  """
  if input_format == 'nlu':
    yield_example_fn = _nlu_examples
  else:
    raise ValueError('Unsupported input_format: {}'.format(input_format))
  for sources, target in yield_example_fn(input_file, domain_name):
    yield sources, target

def _nlu_examples(input_file, target_domain_name=None):
  with tf.gfile.GFile(input_file) as f:
    reader = csv.reader(f)
    session_list = []
    for row_id, (sessionId, raw_query, domain_intent, param) in enumerate(reader):
      query = normal_transformer(raw_query)
      param = normal_transformer(param)
      sources = []
      if row_id > 0 and sessionId == session_list[row_id - 1][0]:
        sources.append(session_list[row_id-1][1]) # last query
      sources.append(query)
      if domain_intent == other_tag:
        domain = other_tag
      else:
        domain, intent = domain_intent.split(".")
      session_list.append((sessionId, query))
      if target_domain_name is not None and target_domain_name != domain:
        continue
      yield sources, (intent,param)


def read_label_map(path):
  """Returns label map read from the given path."""
  with tf.gfile.GFile(path) as f:
    if path.endswith('.json'):
      return json.load(f)
    else:
      label_map = {}
      empty_line_encountered = False
      for tag in f:
        tag = tag.strip()
        if tag:
          label_map[tag] = len(label_map)
        else:
          if empty_line_encountered:
            raise ValueError(
                'There should be no empty lines in the middle of the label map '
                'file.'
            )
          empty_line_encountered = True
      return label_map
