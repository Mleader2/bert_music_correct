# -*- coding: utf-8 -*-
# 混淆词提取算法 不考虑多音字,只考虑单音，故只处理同音字的情况
from pypinyin import lazy_pinyin, pinyin, Style
from Levenshtein import distance # python-Levenshtein
# 编辑距离 (Levenshtein Distance算法)
import copy
import sys
import os, json
from find_entity.acmation import neibu_folder
from curLine_file import curLine


# 给的实体库
entity_folder = "/home/cloudminds/Mywork/corpus/compe/69/slot-dictionaries"
number_map = {"0":"十", "1":"一", "2":"二", "3":"三", "4":"四", "5":"五", "6":"六", "7":"七", "8":"八", "9":"九"}
cons_table = {'n': 'l', 'l': 'n', 'f': 'h', 'h': 'f', 'zh': 'z', 'z': 'zh', 'c': 'ch', 'ch': 'c', 's': 'sh', 'sh': 's'}
vowe_table = {'ai': 'ei', 'an': 'ang', 'en': 'eng', 'in': 'ing',
              'ei': 'ai', 'ang': 'an', 'eng': 'en', 'ing': 'in'}
stop_chars = ["的", "了", "啦", "着","们", "呀", "啊", "地", "呢", ".", ",", "，", "。", "－", "_,", "(", ")", "(", "在", "儿"] # TODO 儿

# 非标准拼音　http://www.pinyin.info/rules/initials_finals.html
w_pinyin_map = {"wu":"u", "wa":"ua", "wo":"uo", "wai":"uai", "wei":"ui", "wan":"uan", "wang":"uang", "wen":"un", "weng":"ueng"}
y_pinyin_map = {"yi":"i", "ya":"ia", "ye":"ie", "yao":"iao", "you":"iu", "yan":"ian", "yang":"iang", "yin":"in","ying":"ing", "yong":"iong"}
def my_lazy_pinyin(word, style=Style.NORMAL, errors='default', strict=True):
    rtn_list = lazy_pinyin(word, style=style, errors=errors, strict=strict)
    # 非标准拼音
    if style == Style.NORMAL:  # 返回整体拼音的列表
        for id, total_pinyin in enumerate(rtn_list):
            if total_pinyin[0]=="w" and total_pinyin in w_pinyin_map:
                rtn_list[id] = w_pinyin_map[total_pinyin]
            elif total_pinyin[0]=="y" and total_pinyin in y_pinyin_map:
                rtn_list[id] = y_pinyin_map[total_pinyin]
    elif style == Style.INITIALS:  # 只返回声母的列表
        total_pinyin_list = lazy_pinyin(word, style=Style.NORMAL, errors=errors, strict=strict)
        for id, (shengmu, total_pinyin) in enumerate(zip(rtn_list, total_pinyin_list)):
            if total_pinyin[0]=="w" and total_pinyin in w_pinyin_map:
                rtn_list[id] = ""
            elif total_pinyin[0]=="y" and total_pinyin in y_pinyin_map:
                rtn_list[id] = ""
    else:
        raise Exception("wrong style")
    return rtn_list


def add_func(entity_info_dict, entity_type, raw_entity, entity_variation, char_distance):
    shengmu_list, yunmu_list, pinyin_list = my_pinyin(word=entity_variation)
    assert len(shengmu_list) == len(yunmu_list) == len(pinyin_list)
    combination = "".join(pinyin_list)
    # if len(combination) < 3:
    #     print(curLine(), "warning:", raw_entity, "combination:", combination)
    entity_info_dict[raw_entity]["combination_list"].append([combination, 0, char_distance, entity_variation])
    base_distance = 1.0  # 修改一个声母或韵母导致的距离，小于１
    for shengmu_index, shengmu in enumerate(shengmu_list):
        if shengmu not in cons_table:  # 没有混淆对
            continue
        shengmu_list_variation = copy.copy(shengmu_list)
        shengmu_list_variation[shengmu_index] = cons_table[shengmu]  # 修改一个声母
        combination_variation = "".join(
            ["%s%s" % (shengmu, yunmu) for shengmu, yunmu in zip(shengmu_list_variation, yunmu_list)])
        entity_info_dict[raw_entity]["combination_list"].append([combination_variation, base_distance, char_distance, entity_variation])
    if entity_type in ["song"]:
        for yunmu_index, yunmu in enumerate(yunmu_list):
            if yunmu not in vowe_table:  # 没有混淆对
                continue
            yunmu_list_variation = copy.copy(yunmu_list)
            yunmu_list_variation[yunmu_index] = vowe_table[yunmu]  # 修改一个韵母
            combination_variation = "".join(
                ["%s%s" % (shengmu, yunmu) for shengmu, yunmu in zip(shengmu_list, yunmu_list_variation)])
            entity_info_dict[raw_entity]["combination_list"].append([combination_variation, base_distance, char_distance, entity_variation])
    return

def add_pinyin(raw_entity, entity_info_dict, pri, entity_type):
    entity = raw_entity
    for k, v in number_map.items():
        entity = entity.replace(k, v)
    if raw_entity not in entity_info_dict:  # 新的实体
        entity_info_dict[raw_entity] = {"priority": pri, "combination_list": []}
        add_func(entity_info_dict, entity_type, raw_entity, entity_variation=entity, char_distance=0)
        for char in stop_chars:
            if char in entity:
                entity_variation = entity.replace(char, "")
                char_distance = len(entity) - len(entity_variation)
                add_func(entity_info_dict, entity_type, raw_entity, entity_variation, char_distance=char_distance)
    else:
        old_pri = entity_info_dict[raw_entity]["priority"]
        if pri > old_pri:
            entity_info_dict[raw_entity]["priority"] = pri
    return

def get_entityType_pinyin(entity_type):
    entity_info_dict = {}
    entity_file = os.path.join(entity_folder, "%s.txt" % entity_type)
    with open(entity_file, "r") as fr:
        lines = fr.readlines()
    priority = 3
    if entity_type in ["song"]:
        priority -= 0.5
    print(curLine(), "get %d %s from %s, priority=%f" % (len(lines), entity_type, entity_file, priority))
    for line in lines:
        raw_entity = line.strip()
        add_pinyin(raw_entity, entity_info_dict, priority, entity_type)

    ### TODO  从标注语料中挖掘得到
    entity_file = os.path.join(neibu_folder, "%s.json" % entity_type)
    with open(entity_file, "r") as fr:
        current_entity_dict = json.load(fr)
    print(curLine(), "get %d %s from %s, priority=%f" % (len(current_entity_dict), entity_type, entity_file, priority))
    for entity_before, entity_after_times in current_entity_dict.items():
        entity_after = entity_after_times[0]
        priority = 4
        if entity_type in ["song"]:
            priority -= 0.5
        add_pinyin(entity_after, entity_info_dict, priority, entity_type)
    return entity_info_dict

# 用编辑距离度量拼音字符串之间的相似度  不考虑相似实体的多音情况
def pinyin_similar_word_danyin(entity_info_dict, word, jichu_distance=0.05, char_ratio=0.55, char_distance=1.0):
    if word in entity_info_dict: # 存在实体，无需纠错
        return 1.0, word
    best_similar_word = None
    top_similar_score = 0
    if "0" in word: # 这里先尝试替换成零，后面会尝试替换成十的情况
        word_ling = word.replace("0", "零")
        top_similar_score_ling, best_similar_word_ling = pinyin_similar_word_danyin(
            entity_info_dict, word_ling, jichu_distance, char_ratio, char_distance)
        if top_similar_score_ling >= 0.99999:
            return top_similar_score_ling, best_similar_word_ling
        elif top_similar_score_ling >= top_similar_score:
            top_similar_score = top_similar_score_ling
            best_similar_word = best_similar_word_ling
    for k,v in number_map.items():
        word = word.replace(k, v)
        if word in entity_info_dict:  # 存在实体，无需纠错
            return 1.0, word
    all_combination = [["".join(my_lazy_pinyin(word)), 0, 0, word]]
    for char in stop_chars:
        if char in word:
            entity_variation = word.replace(char, "")
            c = (len(word) - len(entity_variation)) *char_distance  # TODO
            all_combination.append( ["".join(my_lazy_pinyin(entity_variation)), 0, c, entity_variation] )

    for current_combination, basebase, c1, entity_variation in all_combination: # 当前的各种发音
        if len(current_combination) == 0:
            continue
        similar_word = None
        current_distance = sys.maxsize
        for entity, priority_comList in entity_info_dict.items():
            priority = priority_comList["priority"]
            com_list = priority_comList["combination_list"]
            for com, a, c, entity_variation_ku in com_list:
                d = basebase + jichu_distance*a + char_distance*c+c1 - priority*0.01 + \
                    distance(com, current_combination)*(1.0-char_ratio) + distance(entity_variation_ku,entity_variation) * char_ratio
                if d < current_distance:
                    current_distance = d
                    similar_word = entity
        current_similar_score = 1.0 - float(current_distance) / len(current_combination)
        if current_similar_score > top_similar_score:
            best_similar_word = similar_word
            top_similar_score = current_similar_score
    return top_similar_score, best_similar_word

# 自己包装的函数，返回字符的声母（可能为空，如啊呀），韵母，整体拼音
def my_pinyin(word):
    shengmu_list = my_lazy_pinyin(word, style=Style.INITIALS, strict=False)
    total_pinyin_list = my_lazy_pinyin(word, errors='default')
    yunmu_list = []
    for shengmu, total_pinyin in zip(shengmu_list, total_pinyin_list):
        yunmu = total_pinyin[total_pinyin.index(shengmu) + len(shengmu):]
        yunmu_list.append(yunmu)
    return shengmu_list, yunmu_list, total_pinyin_list


singer_pinyin = get_entityType_pinyin(entity_type="singer")
print(curLine(), len(singer_pinyin), "singer_pinyin")

song_pinyin = get_entityType_pinyin(entity_type="song")
print(curLine(), len(song_pinyin), "song_pinyin")

def correct_song(entity_before,jichu_distance=0.001,   char_ratio=0.53, char_distance=0.1):  # char_ratio=0.48):
    top_similar_score, best_similar_word = pinyin_similar_word_danyin(
        song_pinyin, entity_before, jichu_distance, char_ratio, char_distance)
    return top_similar_score, best_similar_word

def correct_singer(entity_before,jichu_distance=0.001, char_ratio=0.5, char_distance=0.1):  # char_ratio=0.1, char_distance=1.0):
    top_similar_score, best_similar_word = pinyin_similar_word_danyin(
        singer_pinyin, entity_before, jichu_distance, char_ratio, char_distance)
    return top_similar_score, best_similar_word



def test(entity_type, pinyin_ku, score_threshold, jichu_distance, char_ratio=0.0, char_distance=0.1):  # 测试纠错的准确率
    # 把给的实体库加到集合中
    entity_set = set()
    entity_file = os.path.join(entity_folder, "%s.txt" % entity_type)
    with open(entity_file, "r") as fr:
        lines = fr.readlines()
    for line in lines:
        raw_entity = line.strip()
        entity_set.add(raw_entity)
    # 抽取的实体作为测试集合
    entity_file = os.path.join(neibu_folder, "%s.json" % entity_type)
    with open(entity_file, "r") as fr:
        current_entity_dict = json.load(fr)
    test_num = 0.0
    wrong_num = 0.0
    right_num = 0.0
    for entity_before, entity_after_times in current_entity_dict.items():
        entity_after = entity_after_times[0]
        test_num += 1
        top_similar_score, best_similar_word = pinyin_similar_word_danyin(pinyin_ku, entity_before, jichu_distance, char_ratio, char_distance)
        predict_after = entity_before
        if top_similar_score > score_threshold:
            predict_after = best_similar_word
        if predict_after != entity_after:
            wrong_num += 1
        elif entity_before != predict_after: # 纠正对了
            right_num += 1
    return wrong_num, test_num, right_num


if __name__ == "__main__":
    for word in ["霍元甲", "梁梁"]:
        a = correct_song(entity_before=word, jichu_distance=1.0, char_ratio=0.48)
        print(curLine(), word, a)
    for word in ["霍元甲", "m c天佑的歌曲", "v.a"]:
        a = correct_singer(entity_before=word, jichu_distance=1.0, char_ratio=0.1)
        print(curLine(), word, a)

    s = "啊饿恩昂你为什*.\"123c单身"
    s = "啊饿恩昂你为什单身的万瓦呢？"
    s = "打扰一样呆大衣虽四有挖"
    strict = False
    # yunmu_list = my_lazy_pinyin(s, style=Style.FINALS, strict=strict) # 根据声母（拼音的前半部分）得到韵母
    shengmu_list = my_lazy_pinyin(s, style=Style.INITIALS, strict=strict) # 返回声母，但是严格按照标准（strict=True）有些字没有声母会返回空字符串，strict=False时会返回
    pinyin_list = my_lazy_pinyin(s, errors='default') # ""ignore")
    print(curLine(), len(shengmu_list), "shengmu_list:", shengmu_list)
    # print(curLine(), len(yunmu_list), "yunmu_list:", yunmu_list)
    print(curLine(), len(pinyin_list), "pinyin_list:", pinyin_list)
    print(curLine(), my_pinyin(s))
    for id, c in enumerate(s):
        print(curLine(), id, c, my_pinyin(c))



    best_score = -sys.maxsize
    pinyin_ku = None



    # # # # song
    entity_type = "song"
    if entity_type == "song":
        pinyin_ku = song_pinyin
    elif entity_type == "singer":
        pinyin_ku = singer_pinyin
    for char_distance in [0.1]:
        for char_ratio in [0.4, 0.53, 0.55, 0.68]: #
            for score_threshold in  [0.45, 0.65]: #
                for jichu_distance in [0.001]:  # [0.005, 0.01, 0.03, 0.05]:#
                    wrong_num, test_num, right_num = test(entity_type, pinyin_ku, score_threshold, jichu_distance, char_ratio, char_distance=char_distance)
                    score = 0.01*right_num - wrong_num
                    print("char_distance=%f, threshold=%f, jichu_distance=%f, char_ratio=%f, wrong_num=%d, right_num=%d, score=%f" %
                          (char_distance, score_threshold, jichu_distance, char_ratio, wrong_num, right_num, score))
                    if score > best_score:
                        print(curLine(), "score=%f, best_score=%f" % (score, best_score))
                        best_score = score

    # # singer
    # entity_type = "singer" #　
    # if entity_type == "song":
    #     pinyin_ku = song_pinyin
    # elif entity_type == "singer":
    #     pinyin_ku = singer_pinyin
    # for char_distance in [0.05, 0.1]:
    #     for char_ratio in [0.5]:
    #         for score_threshold in  [0.45, 0.65]:
    #             for jichu_distance in [0.001]:  # [0.005, 0.01, 0.03, 0.05]:#
    #                 wrong_num, test_num, right_num = test(entity_type, pinyin_ku, score_threshold, jichu_distance, char_ratio, char_distance)
    #                 score = 0.01*right_num - wrong_num
    #                 print("char_distance=%f, threshold=%f, jichu_distance=%f, char_ratio=%f, wrong_num=%d, right_num=%d, score=%f" %
    #                       (char_distance, score_threshold, jichu_distance, char_ratio, wrong_num, right_num, score))
    #                 if score > best_score:
    #                     print(curLine(), "score=%f, best_score=%f" % (score, best_score))
    #                     best_score = score

