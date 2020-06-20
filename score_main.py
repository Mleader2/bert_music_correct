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
"""Calculates evaluation scores for a prediction TSV file.

The prediction file is produced by predict_main.py and should contain 3 or more
columns:
  1: sources (concatenated)
  2: prediction
  3-n: targets (1 or more)
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app, flags, logging
import score_lib
from bert import tokenization
from curLine_file import curLine

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'prediction_file', None,
    'TSV file containing source, prediction, and target columns.')
flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether score computation should be case insensitive (in the LaserTagger '
    'paper this was set to True).')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_string(
    'domain_name', None,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('prediction_file')
  target_domain_name = FLAGS.domain_name
  print(curLine(), "target_domain_name:", target_domain_name)

  predDomain_list, predIntent_list, domain_list, right_intent_num, right_slot_num, exact_num = score_lib.read_data(
      FLAGS.prediction_file, FLAGS.do_lower_case, target_domain_name=target_domain_name)
  logging.info(f'Read file: {FLAGS.prediction_file}')
  all_num = len(domain_list)
  domain_acc = score_lib.compute_exact_score(predDomain_list, domain_list)
  intent_acc = float(right_intent_num) / all_num
  slot_acc = float(right_slot_num) / all_num
  exact_score = float(exact_num) / all_num
  print('Num=%d, domain_acc=%.4f, intent_acc=%.4f, slot_acc=%.5f, exact_score=%.4f'
        % (all_num, domain_acc, intent_acc, slot_acc, exact_score))


if __name__ == '__main__':
  app.run(main)
