#  发现疑似实体，辅助预测
import os
from collections import defaultdict
import re
import json
from .acmation import KeywordTree, add_to_ac, entity_files_folder, entity_folder
from curLine_file import curLine, normal_transformer

re_phoneNum = re.compile("[0-9一二三四五六七八九十拾]+")  # 编译

# AC自动机, similar to trie tree
# 也许直接读取下载的ｘｌｓ文件更方便，但那样需要安装ｘｌｒｄ模块

domain2entity_map = {}
domain2entity_map["music"] = ["age", "singer", "song", "toplist", "theme", "style", "scene", "language", "emotion", "instrument"]
# domain2entity_map["navigation"] = ["custom_destination", "city"] # "destination", "origin"]
# domain2entity_map["phone_call"] = ["phone_num", "contact_name"]
self_entity_trie_tree = {}  # 总的实体字典  自己建立的某些实体类型的实体树
for domain, entity_type_list in domain2entity_map.items():
    print(curLine(), domain, entity_type_list)
    for entity_type in entity_type_list:
        if entity_type not in self_entity_trie_tree:
            ac = KeywordTree(case_insensitive=True)
        else:
            ac = self_entity_trie_tree[entity_type]

        ### 从标注语料中挖掘得到
        entity_file = os.path.join(entity_files_folder, "%s.json" % entity_type)
        with open(entity_file, "r") as fr:
            current_entity_dict = json.load(fr)
        print(curLine(), "get %d %s from %s" % (len(current_entity_dict), entity_type, entity_file))
        for entity_before, entity_after_times in current_entity_dict.items():
            entity_after = entity_after_times[0]
            pri = 2
            if entity_type in ["song"]:
                pri -= 0.5
            add_to_ac(ac, entity_type, entity_before, entity_after, pri=pri)
        if entity_type == "song":
            add_to_ac(ac, entity_type, "花的画", "花的话", 1.5)

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


def get_slot_info(query, domain):
    useEntityTypeList = domain2entity_map[domain]
    entityTypeMap = get_all_entity(query, useEntityTypeList=useEntityTypeList)
    entity_list_all = []  # 汇总所有实体
    for entity_type, entity_list in entityTypeMap.items():
        for entity in entity_list:
            entity_before = entity['before']
            ignore_flag = False
            if entity_type != "song" and len(entity_before) < 2 and entity_before not in ["家","妈"]:
                ignore_flag = True
            if entity_type == "song" and len(entity_before) < 2 and \
                    entity_before not in {"鱼", "云", "逃", "退", "陶", "美", "图", "默"}:
                ignore_flag = True
            if entity_before in {"什么歌", "一首", "小花","叮当","傻逼", "给你听", "现在", "当我"}:
                ignore_flag = True
            if ignore_flag:
                if entity_before not in "好点没走伤":
                    print(curLine(), "ignore entity_type:%s, entity:%s, query:%s"
                      % (entity_type, entity_before, query))
            else:
                entity_list_all.append((entity_type, entity_before, entity['after'], entity['priority']))
    entity_list_all = sorted(entity_list_all, key=lambda item: len(item[1])*100+item[3],
                             reverse=True)  # new_entity_map 中key是实体,value是实体类型
    slot_info = query
    exist_entityType_set = set()
    replace_mask = [0] * len(query)
    for entity_type, entity_before, entity_after, priority in entity_list_all:
        if entity_before not in query:
            continue
        if entity_type in exist_entityType_set:
            continue  # 已经有这个类型了,忽略 # TODO
        start_location = slot_info.find(entity_before)
        if start_location > -1:
            exist_entityType_set.add(entity_type)
            if entity_after == entity_before:
                entity_info_str = "<%s>%s</%s>" % (entity_type, entity_after, entity_type)
            else:
                entity_info_str = "<%s>%s||%s</%s>" % (entity_type, entity_before, entity_after, entity_type)
            slot_info = slot_info.replace(entity_before, entity_info_str)
            query = query.replace(entity_before, "")
        else:
            print(curLine(), replace_mask, slot_info, "entity_type:", entity_type, entity_before)
    return slot_info

if __name__ == '__main__':
    for query in ["拨打10086", "打电话给100十五", "打电话给一二三拾"]:
        res = get_slot_info(query, domain="phone_call")
        print(curLine(), query, res)

    for query in ["节奏来一首一朵鲜花送给心上人", "陈瑞的歌曲", "放一首销愁"]:
        res = get_slot_info(query, domain="music")
        print(curLine(), query, res)