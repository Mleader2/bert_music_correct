# coding=utf-8
# 识别槽位

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Levenshtein import distance
from curLine_file import curLine
from find_entity.exacter_acmation import get_all_entity
from confusion_words_danyin import  correct_song, correct_singer
tongyin_yuzhi = 0.8
# tongyin_yuzhi_singer = 0.9

# from confusion_words import pinyin_similar_word_noduoyin, singer_pinyin, song_pinyin
# tongyin_yuzhi = 0.75


char_distance_yuzhi = 0.6


# 得到两个字符串在字符级别的相似度　越大越相似
def get_char_similarScore(predict_entity, known_entity):
  '''
  :param predict_entity: 识别得到的槽值
  :param known_entity: 库中已有的实体，用这个实体的长度对编辑距离进行归一化
  :return: 对编辑距离归一化后的分数，越大则越相似
  '''
  d = distance(predict_entity, known_entity)
  score = 1.0 - float(d) / (len(known_entity)+ len(predict_entity))
  if score < char_distance_yuzhi:
    score = 0
  return score


def get_slot_info_str_forMusic(slot_info, raw_query, entityTypeMap):  # 列表连接成字符串
  # 没有直接ｊｏｉｎ得到字符串，因为想要剔除某些只有一个字的实体　例如phone_num
  # print(curLine(), len(slot_info), "slot_info:", slot_info)
  if "货源" in raw_query or "开花" in raw_query:
    print(curLine(), raw_query, entityTypeMap)
  slot_info_block = []
  param_list = []
  current_entityType = None
  for token in slot_info:
    if "<" in token and ">" in token and "/" not in token:  # 一个槽位的开始
      if len(slot_info_block) > 0:
        print(curLine(), len(slot_info), "slot_info:", slot_info)
      assert len(slot_info_block) == 0, "".join(slot_info_block)
      slot_info_block = []
      current_entityType = token[1:-1]
    elif "<" in token and ">" in token and "/" in token:  # 一个槽位的结束
      slot_info_block_str = "".join(slot_info_block)
      # 在循环前按照没找到字符角度相似的实体的情况初始化
      entity_before = slot_info_block_str # 如果没找到字符角度相似的实体，就返回entity_before
      entity_after = slot_info_block_str
      priority = 0  # 优先级
      ignore_flag = False  # 是否忽略这个槽值
      if slot_info_block_str in {"什么歌","一首","小花","叮当","傻逼", "给你听","现在","喜欢","ye ye","没", "j c"}: # 黑名单　不是一个槽值
        ignore_flag = True  # 忽略这个槽值
      else:
        # 逐个实体计算和slot_info_block_str的相似度
        char_similarScore = 0  # 识别的槽值与库中实体在字符上的相似度
        for entity_info in entityTypeMap[current_entityType]:
          current_entity_before = entity_info['before']
          if slot_info_block_str == current_entity_before:  # 在实体库中
            entity_before = current_entity_before
            entity_after = entity_info['after']
            priority = entity_info['priority']
            char_similarScore = 1.0  # 完全匹配，槽值在库中
            if slot_info_block_str != entity_after: # 已经出现过的纠错词
              slot_info_block_str = "%s||%s" % (slot_info_block_str, entity_after)
            break
          current_char_similarScore = get_char_similarScore(predict_entity=slot_info_block_str, known_entity=current_entity_before)
          # print(curLine(), slot_info_block_str, current_entity_before, current_char_similarScore)
          if current_char_similarScore > char_similarScore:  # 在句子找到字面上更接近的实体
            entity_before = current_entity_before
            entity_after = entity_info['after']
            priority = entity_info['priority']
            char_similarScore = current_char_similarScore
        if priority == 0:
          if current_entityType not in {"singer", "song"}:
            ignore_flag = True
          if len(entity_before) < 2:   # 忽略这个短的槽值
            ignore_flag = True
          if current_entityType == "song" and entity_before in {"鱼", "云", "逃", "退", "陶", "美", "图", "默", "哭", "雪"}:  # TODO ｓｏｎｇ的白名单, 来一首不
            ignore_flag = False  # 这个要在后面判断
      if ignore_flag:
        if entity_before not in "好点没走":
          print(curLine(), raw_query, "ignore entityType:%s, slot:%s, entity_before:%s" % (current_entityType, slot_info_block_str, entity_before))
      else:
        param_list.append({'entityType':current_entityType, 'before':entity_before, 'after':entity_after, 'priority':priority})
      slot_info_block = []
      current_entityType = None
    elif current_entityType is not None: # is in a slot
      slot_info_block.append(token)
    # print(curLine(), token, len(param_list), "param_list:", param_list)
  param_list_sorted = sorted(param_list, key=lambda item: len(item['before'])*100+item['priority'],
                             reverse=True)
  slot_info_str = raw_query
  replace_query = raw_query
  for param in param_list_sorted:
    entity_before = param['before']
    replace_str = entity_before
    if replace_str not in replace_query:
      # print(curLine(), "%s not in %s" % (replace_str, replace_query))
      # input(curLine())
      continue
    replace_query = replace_query.replace(replace_str, "")
    entityType = param['entityType']

    if param['priority'] == 0:  # 模型识别的结果不在库中，尝试用拼音纠错
      similar_score = 0.0
      best_similar_word = None
      if entityType == "singer":
        similar_score, best_similar_word = correct_singer(entity_before, jichu_distance=0.001, char_ratio=0.1)
        similar_score -= 0.1 # TODO 提高对ｓｉｎｇｅｒ的阈值
      elif entityType == "song":
        similar_score, best_similar_word = correct_song(entity_before, jichu_distance=0.001, char_ratio=0.48)

      # if entityType == "singer":
      #   similar_score, best_similar_word = pinyin_similar_word_noduoyin(singer_pinyin, entity_before)
      # elif entityType == "song":
      #   similar_score, best_similar_word = pinyin_similar_word_noduoyin(song_pinyin, entity_before)

      if similar_score > tongyin_yuzhi and best_similar_word != entity_before:
          print(curLine(), entityType, "entity_before:",entity_before, best_similar_word, similar_score, "\n")
          param['after'] = best_similar_word


      # if similar_score > tongyin_yuzhi and best_similar_word != entity_before:
      #     # print(curLine(), entityType, "entity_before:",entity_before, best_similar_word, similar_score, "\n")
      #     param['after'] = best_similar_word

    if entity_before != param['after']:
      replace_str = "%s||%s" % (entity_before, param['after'])
    assert entity_before in slot_info_str
    slot_info_str = slot_info_str.replace(entity_before, "<%s>%s</%s>" % (entityType, replace_str, entityType))
  # print(curLine(), "slot_info_str:", slot_info_str)
  # print(curLine(), len(param_list_sorted), "param_list_sorted:", param_list_sorted)
  # print(curLine())
  return slot_info_str


def get_slot_info_str(slot_info, raw_query, entityTypeMap): # 列表连接成字符串
  # 没有直接ｊｏｉｎ得到字符串，因为想要剔除某些只有一个字的实体　例如phone_num
  # print(curLine(), len(slot_info), "slot_info:", slot_info)
  if "货源" in raw_query or "开花" in raw_query:
    print(curLine(), raw_query, entityTypeMap)
  slot_info_str = []
  slot_info_block = []
  current_entityType = None
  for token in slot_info:
    if "<" in token and ">" in token and "/" not in token:  # 一个槽位的开始
      if len(slot_info_block) > 0:
        print(curLine(), len(slot_info), "slot_info:", slot_info)
      assert len(slot_info_block) == 0, "".join(slot_info_block)
      slot_info_block = []
      current_entityType = token[1:-1]
    elif "<" in token and ">" in token and "/" in token:  # 一个槽位的结束
      ignore_flag = False
      slot_info_block_str = "".join(slot_info_block)
      if (current_entityType != "phone_num" or len(slot_info_block_str) > 1) and \
              slot_info_block_str not in {"什么歌", "一首", "小花","叮当","傻逼", "给你听","现在"}: # 确认是一个槽值 # TODO 为什么phone_num例外
        if current_entityType in ["singer", "song"]:  # 同音词查找
          known_entity_flag = False  # 假设预测的实体不在实体库中
          for entity_info in entityTypeMap[current_entityType]:
            if slot_info_block_str == entity_info['before']: # 在实体库中
              entity_after = entity_info['after']
              if slot_info_block_str != entity_after: # 已经出现过的纠错词
                slot_info_block_str = "%s||%s" % (slot_info_block_str, entity_after)
                known_entity_flag = True
              break

          if not known_entity_flag:# 预测的实体不在实体库中,则要尝试结合拼音进行纠错
            # TODO 目前未考虑多个纠错后的after具有相同拼音的情况　可以结合音调和字符进行筛选
            if len(slot_info_block_str) < 2 and slot_info_block_str not in {"鱼","云","逃","退"}:
              ignore_flag = True  # 忽略一个字的
            elif slot_info_block_str in {"什么歌", "一首"}:
              ignore_flag = True # 忽略一个字的
            else:
              if current_entityType == "singer":
                similar_score, best_similar_word = correct_singer(slot_info_block_str)
              elif current_entityType == "song":
                similar_score, best_similar_word = correct_song(slot_info_block_str)
              if best_similar_word != slot_info_block_str:
                if similar_score > tongyin_yuzhi:
                  # print(curLine(), current_entityType, slot_info_block_str, best_similar_word, similar_score)
                  slot_info_block_str = "%s||%s" % (slot_info_block_str, best_similar_word)
        elif current_entityType in ["theme", "style", "age", "toplist", "emotion", "language", "instrument", "scene"]:
          entity_info_list = get_all_entity(raw_query, useEntityTypeList=[current_entityType])[current_entityType]
          ignore_flag = True  # 忽略不在库中的实体
          min_distance = 1000
          houxuan = None
          for entity_info in entity_info_list:
            entity_before = entity_info["before"]
            if slot_info_block_str == entity_before:
              houxuan = entity_before
              min_distance = 0
              break
          if houxuan is not None and min_distance <= 1:
            if slot_info_block_str != houxuan:
              print(curLine(), len(entity_info_list), "entity_info_list:", entity_info_list, "slot_info_block_str:",
                    slot_info_block_str)
              print(curLine(), "change %s from %s to %s" % (current_entityType, slot_info_block_str, houxuan))
            slot_info_block_str = houxuan
            ignore_flag = False # 不忽略
        if ignore_flag:
          slot_info_str.extend([slot_info_block_str])
          # print(curLine(), "ignore:",current_entityType, slot_info_block_str)
          # input(curLine())
        else:
          slot_info_str.extend(["<%s>" % current_entityType, slot_info_block_str, token])

      else:  # 忽略这个短的槽值
        print(curLine(), "ignore entityType:%s, slot:%s" % (current_entityType, slot_info_block_str))
        slot_info_str.append(slot_info_block_str)
      current_entityType = None
      slot_info_block = []
    elif current_entityType is None:  # not slot
      slot_info_str.append(token)
    else:  # is in a slot
      slot_info_block.append(token)
  slot_info_str = "".join(slot_info_str)
  # print(curLine(), "slot_info_str:", slot_info_str)
  return slot_info_str