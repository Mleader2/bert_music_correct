#  发现疑似实体，辅助训练
#　用ａｃ自动机构建发现疑似实体的工具
import os
from collections import defaultdict
import json
import re
from .acmation import KeywordTree, add_to_ac, entity_files_folder, entity_folder
from curLine_file import curLine, normal_transformer

domain2entity_map = {}
domain2entity_map["music"] = ["age", "singer", "song", "toplist", "theme", "style", "scene", "language", "emotion", "instrument"]
domain2entity_map["navigation"] = ["custom_destination", "city"]  # place city
domain2entity_map["phone_call"] = ["phone_num", "contact_name"]

re_phoneNum = re.compile("[0-9一二三四五六七八九十拾]+")  # 编译


# 也许直接读取下载的ｘｌｓ文件更方便，但那样需要安装ｘｌｒｄ模块
self_entity_trie_tree = {}  # 总的实体字典  自己建立的某些实体类型的实体树
for domain, entity_type_list in domain2entity_map.items():
    print(curLine(), domain, entity_type_list)
    for entity_type in entity_type_list:
        if entity_type not in self_entity_trie_tree:
            ac = KeywordTree(case_insensitive=True)
        else:
            ac = self_entity_trie_tree[entity_type]

        # TODO
        if entity_type == "city":
            # for current_entity_type in ["city", "province"]:
            #     entity_file = waibu_folder + "%s.json" % current_entity_type
            #     with open(entity_file, "r") as f:
            #         current_entity_dict = json.load(f)
            #         print(curLine(), "get %d %s from %s" %
            #               (len(current_entity_dict), current_entity_type, entity_file))
            #     for entity_before, entity_times in current_entity_dict.items():
            #         entity_after = entity_before
            #         add_to_ac(ac, entity_type, entity_before, entity_after, pri=1)

            ## 从标注语料中挖掘得到的地名
            for current_entity_type in ["destination", "origin"]:
                entity_file = os.path.join(entity_files_folder, "%s.json" % current_entity_type)
                with open(entity_file, "r") as f:
                    current_entity_dict = json.load(f)
                    print(curLine(), "get %d %s from %s" %
                          (len(current_entity_dict), current_entity_type, entity_file))
                for entity_before, entity_after_times in current_entity_dict.items():
                    entity_after = entity_after_times[0]
                    add_to_ac(ac, entity_type, entity_before, entity_after, pri=2)
                input(curLine())

        # 给的实体库,最高优先级
        entity_file = os.path.join(entity_folder, "%s.txt" % entity_type)
        if os.path.exists(entity_file):
            with open(entity_file, "r") as fr:
                lines = fr.readlines()
            print(curLine(), "get %d %s from %s" % (len(lines), entity_type, entity_file))
            for line in lines:
                entity_after = line.strip()
                entity_before = entity_after # TODO
                pri = 3
                if entity_type in ["song"]:
                    pri -= 0.5
                add_to_ac(ac, entity_type, entity_before, entity_after, pri=pri)
        ac.finalize()
        self_entity_trie_tree[entity_type] = ac


def get_all_entity(corpus, useEntityTypeList):
    self_entityTypeMap = defaultdict(list)
    for entity_type in useEntityTypeList:
        result = self_entity_trie_tree[entity_type].search(corpus)
        for res in result:
            after, priority = res.meta_data
            self_entityTypeMap[entity_type].append({'before': res.keyword, 'after': after, "priority":priority})
    if "phone_num" in useEntityTypeList:
        token_numbers = re_phoneNum.findall(corpus)
        for number in token_numbers:
            self_entityTypeMap["phone_num"].append({'before':number, 'after':number, 'priority': 2})
    return self_entityTypeMap