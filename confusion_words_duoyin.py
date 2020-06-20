# -*- coding: utf-8 -*-
# 混淆词提取算法
from pypinyin import lazy_pinyin, pinyin, Style
# pinyin 的方法默认带声调，而 lazy_pinyin 方法不带声调,但是会得到不常见的多音字
# pinyin(c, heteronym=True, style=0)  不考虑音调情况下的多音字

# from tqdm import tqdm
import json
import jieba
from Levenshtein import distance # python-Levenshtein
# 编辑距离 (Levenshtein Distance算法)
import os

# import re
# import dimsim
import time
import copy
from curLine_file import curLine


# 给的实体库
entity_folder = "/home/cloudminds/Mywork/corpus/compe/69/slot-dictionaries"
number_map = {"0":"零", "1":"一", "2":"二", "3":"三", "4":"四", "5":"五", "6":"六", "7":"七", "8":"八", "9":"九"}
def get_pinyin_combination(entity):
    all_combination = [""]
    for zi in entity:
        if zi in number_map:
            zi = number_map[zi]
        zi_pinyin_list = pinyin(zi, heteronym=True, style=0)[0]
        pre_combination = copy.deepcopy(all_combination)
        all_combination = []
        for index, zi_pinyin in enumerate(zi_pinyin_list): # 逐个添加每种发音
            for com in pre_combination:  # com代表一种情况
                # com.append(zi_pinyin)
                # new = com + [zi_pinyin]
                # new = "".join(new)
                all_combination.append("%s%s" % (com, zi_pinyin)) # TODO  要不要加分隔符
    return all_combination

def get_entityType_pinyin(entity_type):
    entity_info_dict = {}
    entity_file = os.path.join(entity_folder, "%s.txt" % entity_type)
    with open(entity_file, "r") as fr:
        lines = fr.readlines()
    pri = 3
    if entity_type in ["song"]:
        pri -= 0.5
    print(curLine(), "get %d %s from %s, pri=%f" % (len(lines), entity_type, entity_file, pri))
    for line in lines:
        entity_after = line.strip()
        all_combination = get_pinyin_combination(entity=entity_after)
        # for combination in all_combination:
        if entity_after not in entity_info_dict:  # 新的发音
            entity_info_dict[entity_after] = (all_combination, pri)
        else: # 一般不会重复，最多重复一次
            _, old_pri = entity_info_dict[entity_after]
            if pri > old_pri:
                entity_info_dict[entity_after] = (all_combination, pri)
    return entity_info_dict

# 用编辑距离度量拼音字符串之间的相似度
def pinyin_similar_word(entity_info_dict, word):
    similar_word = None
    if word in entity_info_dict: # 存在实体，无需纠错
        return 0, word
    all_combination = get_pinyin_combination(entity=word)
    top_similar_score = 0
    for current_combination in all_combination: # 当前的各种发音
        current_distance = 10000

        for entity_after,(com_list, pri) in entity_info_dict.items():
            for com in com_list:
                d = distance(com, current_combination)
                if d < current_distance:
                    current_distance = d
                    similar_word = entity_after
        current_similar_score = 1.0 - float(current_distance) / len(current_combination)
        print(curLine(), "current_combination:%s, %f" % (current_combination, current_similar_score), similar_word, current_distance)



singer_pinyin = get_entityType_pinyin(entity_type="singer")
print(curLine(), len(singer_pinyin), "singer_pinyin")

song_pinyin = get_entityType_pinyin(entity_type="song")
print(curLine(), len(song_pinyin), "song_pinyin")



if __name__ == "__main__":
    s = "你为什123c单身"
    for c in s:
        c_pinyin = lazy_pinyin(s, errors="ignore")
        print(curLine(), c,c_pinyin)
    print(curLine(), pinyin(s, heteronym=True, style=0))
    # # print(pinyin(c, heteronym=True, style=0))

    # all_combination = get_pinyin_combination(entity=s)
    #
    # for index, combination in enumerate(all_combination):
    #     print(curLine(), index, combination)
    # for s in ["abc", "abs1", "fabc"]:
    #     ed = distance("abs", s)
    #     print(s, ed)

    pinyin_similar_word(singer_pinyin, "周杰")
    pinyin_similar_word(singer_pinyin, "前任")
