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
"""Convert a dataset into the TFRecord format.

The resulting TFRecord file will be used when training a LaserTagger model.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Text

from absl import app
from absl import flags
from absl import logging
from bert import tokenization
import bert_example
import json
import utils

import tensorflow as tf
from find_entity.probable_acmation import domain2entity_map, get_all_entity
from curLine_file import curLine


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples to be converted to '
    'tf.Examples.')
flags.DEFINE_enum(
    'input_format', None, ['nlu'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string('output_tfrecord_train', None,
                    'Path to the resulting TFRecord file.')
flags.DEFINE_string('output_tfrecord_dev', None,
                    'Path to the resulting TFRecord file.')
flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per line.')
flags.DEFINE_string(
    'slot_label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per line.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_string(
    'domain_name', None, 'domain_name')
flags.DEFINE_string(
    'entity_type_list_file', None, 'path of entity_type_list_file')

def _write_example_count(count: int, output_file: str) -> Text:
    """Saves the number of converted examples to a file.

    This count is used when determining the number of training steps.

    Args:
      count: The number of converted examples.

    Returns:
      The filename to which the count is saved.
    """
    count_fname = output_file + '.num_examples.txt'
    with tf.gfile.GFile(count_fname, 'w') as count_writer:
        count_writer.write(str(count))
    return count_fname



def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_tfrecord_train')
    flags.mark_flag_as_required('output_tfrecord_dev')
    flags.mark_flag_as_required('vocab_file')
    target_domain_name = FLAGS.domain_name
    entity_type_list = domain2entity_map[target_domain_name]

    print(curLine(), "target_domain_name:", target_domain_name, len(entity_type_list), "entity_type_list:", entity_type_list)
    builder = bert_example.BertExampleBuilder({}, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, slot_label_map={},
                                              entity_type_list=entity_type_list, get_entity_func=get_all_entity)

    num_converted = 0
    num_ignored = 0
    # ff = open("new_%s.txt" % target_domain_name, "w") # TODO
    with tf.python_io.TFRecordWriter(FLAGS.output_tfrecord_train) as writer_train:
        for i, (sources, target) in enumerate(utils.yield_sources_and_targets(
                FLAGS.input_file, FLAGS.input_format, target_domain_name)):
            logging.log_every_n(
                logging.INFO,
                f'{i} examples processed, {num_converted} converted to tf.Example.',
                10000)
            if len(sources[0]) > 35:  # 忽略问题太长的样本
                num_ignored += 1
                print(curLine(), "ignore num_ignored=%d, question length=%d" % (num_ignored, len(sources[0])))
                continue
            example1, _, info_str = builder.build_bert_example(sources, target)
            example =example1.to_tf_example().SerializeToString()
            writer_train.write(example)
            num_converted += 1
            # ff.write("%d %s\n" % (i, info_str))
    logging.info(f'Done. {num_converted} examples converted to tf.Example, num_ignored {num_ignored} examples.')
    for output_file in [FLAGS.output_tfrecord_train]:
        count_fname = _write_example_count(num_converted, output_file=output_file)
        logging.info(f'Wrote:\n{output_file}\n{count_fname}')
    with open(FLAGS.label_map_file, "w") as f:
        json.dump(builder._label_map, f, ensure_ascii=False, indent=4)
    print(curLine(), "save %d to %s" % (len(builder._label_map), FLAGS.label_map_file))
    with open(FLAGS.slot_label_map_file, "w") as f:
        json.dump(builder.slot_label_map, f, ensure_ascii=False, indent=4)
    print(curLine(), "save %d to %s" % (len(builder.slot_label_map), FLAGS.slot_label_map_file))

    with open(FLAGS.entity_type_list_file, "w") as f:
        json.dump(domain2entity_map, f, ensure_ascii=False, indent=4)
    print(curLine(), "save %d to %s" % (len(domain2entity_map), FLAGS.entity_type_list_file))


if __name__ == '__main__':
    app.run(main)