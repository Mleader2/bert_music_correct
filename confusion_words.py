# -*- coding: utf-8 -*-
# 混淆词提取算法 不考虑多音字,只处理同音字的情况
from pypinyin import lazy_pinyin, pinyin, Style
# pinyin 的方法默认带声调，而 lazy_pinyin 方法不带声调,但是会得到不常见的多音字
# pinyin(c, heteronym=True, style=0)  不考虑音调情况下的多音字
from Levenshtein import distance # python-Levenshtein
# 编辑距离 (Levenshtein Distance算法)
import os
from curLine_file import curLine
from find_entity.acmation import entity_folder

number_map = {"0":"零", "1":"一", "2":"二", "3":"三", "4":"四", "5":"五", "6":"六", "7":"七", "8":"八", "9":"九"}
cons_table = {'n': 'l', 'l': 'n', 'f': 'h', 'h': 'f', 'zh': 'z', 'z': 'zh', 'c': 'ch', 'ch': 'c', 's': 'sh', 'sh': 's'}
# def get_similar_pinyin(entity_type): # 存的时候直接把拼音的变体也保存下来
#     entity_info_dict = {}
#     entity_file = os.path.join(entity_folder, "%s.txt" % entity_type)
#     with open(entity_file, "r") as fr:
#         lines = fr.readlines()
#     pri = 3
#     if entity_type in ["song"]:
#         pri -= 0.5
#     print(curLine(), "get %d %s from %s, pri=%f" % (len(lines), entity_type, entity_file, pri))
#     for line in lines:
#         entity = line.strip()
#         for k,v in number_map.items():
#             entity.replace(k, v)
#         # for combination in all_combination:
#         if entity not in entity_info_dict:  # 新的实体
#             combination = "".join(lazy_pinyin(entity))  # default:默认行为，不处理，原木原样返回  , errors="ignore"
#             if len(combination) < 2:
#                 print(curLine(), "warning:", entity, "combination:", combination)
#             entity_info_dict[entity] = (combination, pri)

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
        entity = line.strip()
        for k,v in number_map.items():
            entity.replace(k, v)
        # for combination in all_combination:
        if entity not in entity_info_dict:  # 新的实体
            combination = "".join(lazy_pinyin(entity))  # default:默认行为，不处理，原木原样返回  , errors="ignore"
            if len(combination) < 2:
                print(curLine(), "warning:", entity, "combination:", combination)
            entity_info_dict[entity] = (combination, pri)
        else:
            combination, old_pri = entity_info_dict[entity]
            if pri > old_pri:
                entity_info_dict[entity] = (combination, pri)
    return entity_info_dict

# 用编辑距离度量拼音字符串之间的相似度  不考虑相似实体的多音情况
def pinyin_similar_word_noduoyin(entity_info_dict, word):
    if word in entity_info_dict: # 存在实体，无需纠错
        return 1.0, word
    best_similar_word = None
    top_similar_score = 0
    try:
        all_combination = ["".join(lazy_pinyin(word))] #　get_pinyin_combination(entity=word) #
        for current_combination in all_combination: # 当前的各种发音
            if len(current_combination) == 0:
                print(curLine(), "word:", word)
                continue
            similar_word = None
            current_distance = 10000
            for entity,(com, pri) in entity_info_dict.items():
                char_ratio = 0.0
                d = distance(com, current_combination)*(1.0-char_ratio) + distance(entity, word) * char_ratio
                if d < current_distance:
                    current_distance = d
                    similar_word = entity
                # if d<=2.5:
                #     print(curLine(),com, current_combination, distance(com, current_combination), distance(entity, word) )
                #     print(curLine(), word, entity, similar_word, "current_distance=", current_distance)


            current_similar_score = 1.0 - float(current_distance) / len(current_combination)
            # print(curLine(), "current_combination:%s, %f" % (current_combination, current_similar_score), similar_word, current_distance)
            if current_similar_score > top_similar_score:
                # print(curLine(), current_similar_score, top_similar_score, best_similar_word, similar_word)
                best_similar_word = similar_word
                top_similar_score = current_similar_score
    except Exception as error:
        print(curLine(), "error:", error)
    return top_similar_score, best_similar_word

# 自己包装的函数，返回字符的声母（可能为空，如啊呀），韵母，整体拼音
def my_pinyin(char):
    shengmu = pinyin(char, style=Style.INITIALS, strict=True)[0][0]
    yunmu = pinyin(char, style=Style.FINALS, strict=True)[0][0]
    total_pinyin = lazy_pinyin(char, errors='default')[0]
    if shengmu + yunmu != total_pinyin:
        print(curLine(), "char:", char, ",shengmu:%s, yunmu:%s" % (shengmu, yunmu), total_pinyin)
    return shengmu, yunmu, total_pinyin

singer_pinyin = get_entityType_pinyin(entity_type="singer")
print(curLine(), len(singer_pinyin), "singer_pinyin")

song_pinyin = get_entityType_pinyin(entity_type="song")
print(curLine(), len(song_pinyin), "song_pinyin")


if __name__ == "__main__":
    # pinyin('中心', heteronym=True)  # 启用多音字模式


    import confusion_words_duoyin
    res = pinyin_similar_word_noduoyin(song_pinyin, "体验") # confusion_words_duoyin.song_pinyin, "体验")
    print(curLine(), res)

    s = "啊饿恩昂你为什*.\"123c单身"
    s = "啊饿恩昂你为什单身的呢？"
    yunmu_list = pinyin(s, style=Style.FINALS)
    shengmu_list = pinyin(s, style=Style.INITIALS, strict=False) # 返回声母，但是严格按照标准（strict=True）有些字没有声母会返回空字符串，strict=False时会返回
    pinyin_list = lazy_pinyin(s, errors='default') # ""ignore")
    print(curLine(), len(shengmu_list), "shengmu_list:", shengmu_list)
    print(curLine(), len(yunmu_list), "yunmu_list:", yunmu_list)
    print(curLine(), len(pinyin_list), "pinyin_list:", pinyin_list)
    for id, c in enumerate(s):
        print(curLine(), id, c, my_pinyin(c))
        # print(curLine(), c, pinyin(c, style=Style.INITIALS), s)
        # print(curLine(), c,c_pinyin)
    # print(curLine(), pinyin(s, heteronym=True, style=0))
    # # print(pinyin(c, heteronym=True, style=0))

    # all_combination = get_pinyin_combination(entity=s)
    #
    # for index, combination in enumerate(all_combination):
    #     print(curLine(), index, combination)
    # for s in ["abc", "abs1", "fabc"]:
    #     ed = distance("abs", s)
    #     print(s, ed)
    for predict_singer in ["周杰伦", "前任", "朱姐", "奇隆"]:
        top_similar_score, best_similar_word = pinyin_similar_word_noduoyin(singer_pinyin, predict_singer)
        print(curLine(), predict_singer, top_similar_score, best_similar_word)
