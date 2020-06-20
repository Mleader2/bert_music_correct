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
"""Compute realized predictions for a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import math, time
from termcolor import colored
import tensorflow as tf

import bert_example
import predict_utils
import csv
import utils
from find_entity import exacter_acmation
from curLine_file import curLine, normal_transformer, other_tag

sep_str = '$'
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse','qa', 'nlu'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the CSV file where the predictions are written to.')
flags.DEFINE_string(
    'submit_file', None,
    'Path to the CSV file where the predictions are written to.')
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
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')
flags.DEFINE_string(
    'domain_name', None,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_string(
    'entity_type_list_file', None, 'path of entity_type_list_file')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('label_map_file')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('saved_model')
    label_map = utils.read_label_map(FLAGS.label_map_file)
    slot_label_map = utils.read_label_map(FLAGS.slot_label_map_file)
    target_domain_name = FLAGS.domain_name
    print(curLine(), "target_domain_name:", target_domain_name)
    entity_type_list = utils.read_label_map(FLAGS.entity_type_list_file)[FLAGS.domain_name]

    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, slot_label_map=slot_label_map,
                                              entity_type_list=entity_type_list, get_entity_func=exacter_acmation.get_all_entity)
    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map, slot_label_map, target_domain_name=target_domain_name)
    print(colored("%s saved_model:%s" % (curLine(), FLAGS.saved_model), "red"))

    ##### test
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))


    domain_list = []
    slot_info_list = []
    intent_list = []

    predict_domain_list = []
    previous_pred_slot_list = []
    previous_pred_intent_list = []
    sources_list = []
    predict_batch_size = 64
    limit = predict_batch_size * 1500 # 5184 #　10001 #
    with tf.gfile.GFile(FLAGS.input_file) as f:
        reader = csv.reader(f)
        session_list = []
        for row_id, line in enumerate(reader):
            if len(line) == 1:
                line = line[0].strip().split("\t")
            if len(line) > 4:  # 有标注
                (sessionId, raw_query, predDomain, predIntent, predSlot, domain, intent, slot) = line
                domain_list.append(domain)
                intent_list.append(intent)
                slot_info_list.append(slot)
            else:
                (sessionId, raw_query, predDomainIntent, predSlot) = line
                if "." in predDomainIntent:
                    predDomain,predIntent = predDomainIntent.split(".")
                else:
                    predDomain,predIntent = predDomainIntent, predDomainIntent
            if "忘记电话" in raw_query:
                predDomain = "phone_call" # rule
            if "专用道" in raw_query:
                predDomain = "navigation" # rule
            predict_domain_list.append(predDomain)
            previous_pred_slot_list.append(predSlot)
            previous_pred_intent_list.append(predIntent)
            query = normal_transformer(raw_query)
            sources = []
            if row_id > 0 and sessionId == session_list[row_id - 1][0]:
                sources.append(session_list[row_id - 1][1])  # last query
            sources.append(query)
            session_list.append((sessionId, raw_query))
            sources_list.append(sources)

            if len(sources_list) >= limit:
                print(colored("%s stop reading at %d to save time" %(curLine(), limit), "red"))
                break

    number = len(sources_list)  # 总样本数

    predict_intent_list = []
    predict_slot_list = []
    predict_batch_size = min(predict_batch_size, number)
    batch_num = math.ceil(float(number) / predict_batch_size)
    start_time = time.time()
    num_predicted = 0
    modemode = 'a'
    if len(domain_list) > 0:  # 有标注
        modemode = 'w'
    with tf.gfile.Open(FLAGS.output_file, modemode) as writer:
        # if len(domain_list) > 0:  # 有标注
        #     writer.write("\t".join(["sessionId", "query", "predDomain", "predIntent", "predSlot", "domain", "intent", "Slot"]) + "\n")
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            predict_domain_batch = predict_domain_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            predict_intent_batch, predict_slot_batch = predictor.predict_batch(sources_batch=sources_batch, target_domain_name=target_domain_name, predict_domain_batch=predict_domain_batch)
            assert len(predict_intent_batch) == len(sources_batch)
            num_predicted += len(predict_intent_batch)
            for id, [predict_intent, predict_slot_info, sources] in enumerate(zip(predict_intent_batch, predict_slot_batch, sources_batch)):
                sessionId, raw_query = session_list[batch_id * predict_batch_size + id]
                predict_domain = predict_domain_list[batch_id * predict_batch_size + id]
                # if predict_domain == "music":
                #     predict_slot_info = raw_query
                #     if predict_intent == "play":  # 模型分类到播放意图，但没有找到槽位，这时用ａｃ自动机提高召回
                #         predict_intent_rule, predict_slot_info = rules(raw_query, predict_domain, target_domain_name)
                        # # if predict_intent_rule in {"pause", "next"}:
                        # #     predict_intent = predict_intent_rule
                        # if "<" in predict_slot_info_rule : # and "<" not in predict_slot_info:
                        #     predict_slot_info = predict_slot_info_rule
                        #     print(curLine(), "predict_slot_info_rule:", predict_slot_info_rule)
                        #     print(curLine())

                if predict_domain != target_domain_name:  #  不是当前模型的ｄｏｍａｉｎ，用规则识别
                    predict_intent = previous_pred_intent_list[batch_id * predict_batch_size + id]
                    predict_slot_info = previous_pred_slot_list[batch_id * predict_batch_size + id]
                # else:
                #     print(curLine(), predict_intent, "predict_slot_info:", predict_slot_info)
                predict_intent_list.append(predict_intent)
                predict_slot_list.append(predict_slot_info)
                if len(domain_list) > 0:  # 有标注
                    domain = domain_list[batch_id * predict_batch_size + id]
                    intent = intent_list[batch_id * predict_batch_size + id]
                    slot = slot_info_list[batch_id * predict_batch_size + id]
                    domain_flag = "right"
                    if domain != predict_domain:
                        domain_flag = "wrong"
                    writer.write("\t".join([sessionId, raw_query, predict_domain, predict_intent, predict_slot_info, domain, intent, slot]) + "\n") # , domain_flag
            if batch_id % 5 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    print(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time/num_predicted*60} s.')


    if FLAGS.submit_file is not None:
        import collections, os
        domain_counter = collections.Counter()
        if os.path.exists(path=FLAGS.submit_file):
            os.remove(FLAGS.submit_file)
        with open(FLAGS.submit_file, 'w',encoding='UTF-8') as f:
            writer = csv.writer(f, dialect='excel')
            # writer.writerow(["session_id", "query", "intent", "slot_annotation"])  # TODO
            for example_id, sources in enumerate(sources_list):
                sessionId, raw_query = session_list[example_id]
                predict_domain = predict_domain_list[example_id]
                predict_intent = predict_intent_list[example_id]
                predict_domain_intent = other_tag
                domain_counter.update([predict_domain])
                slot = raw_query
                if predict_domain != other_tag:
                    predict_domain_intent = "%s.%s" % (predict_domain, predict_intent)
                    slot = predict_slot_list[example_id]
                # if predict_domain == "navigation": # TODO  TODO
                #     predict_domain_intent = other_tag
                #     slot = raw_query
                line = [sessionId, raw_query, predict_domain_intent, slot]
                writer.writerow(line)
        print(curLine(), "example_id=", example_id)
        print(curLine(), "domain_counter:", domain_counter)
        cost_time = (time.time() - start_time) / 60.0
        num_predicted = example_id+1
        print(curLine(), "%s cost %f s" % (target_domain_name, cost_time))
        print(
            f'{curLine()} {num_predicted} predictions saved to:{FLAGS.submit_file}, cost {cost_time} min, ave {cost_time/num_predicted*60} s.')



def rules(raw_query, predict_domain, target_domain_name):
    predict_intent = predict_domain  # OTHERS
    slot_info = raw_query
    if predict_domain == "navigation":
        predict_intent = 'navigation'
        if "打开" in raw_query:
            predict_intent = "open"
        elif "开始" in raw_query:
            predict_intent = "start_navigation"
        for word in predict_utils.cancel_keywords:
            if word in raw_query:
                predict_intent = 'cancel_navigation'
                break
        # slot_info = raw_query
        # if predict_intent == 'navigation': TODO
        slot_info = exacter_acmation.get_slot_info(raw_query, domain=predict_domain)
        # if predict_intent != 'navigation': # TODO
        #     print(curLine(), "slot_info:", slot_info)
    elif predict_domain == 'music':
        predict_intent = 'play'
        for word in predict_utils.cancel_keywords:
            if word in raw_query:
                predict_intent = 'pause'
                break
        for word in ["下一", "换一首", "换一曲", "切歌", "其他歌"]:
            if word in raw_query:
                predict_intent = 'next'
                break
        slot_info = exacter_acmation.get_slot_info(raw_query, domain=predict_domain)
        if predict_intent not in ['play','pause'] and slot_info != raw_query: # 根据槽位修改意图　　换一首<singer>高安</singer>的<song>红尘情歌</song>
            print(curLine(), predict_intent, slot_info)
            predict_intent = 'play'
        # if predict_intent != 'play': # 换一首<singer>高安</singer>的<song>红尘情歌</song>
        #     print(curLine(), predict_intent, slot_info)
    elif predict_domain == 'phone_call':
        predict_intent = 'make_a_phone_call'
        for word in predict_utils.cancel_keywords:
            if word in raw_query:
                predict_intent = 'cancel'
                break
        slot_info = exacter_acmation.get_slot_info(raw_query, domain=predict_domain)
    return predict_intent, slot_info

if __name__ == '__main__':
    app.run(main)
