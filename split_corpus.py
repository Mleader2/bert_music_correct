# coding=utf-8
# 分割语料
import os
import csv, json
from collections import defaultdict
import re
from curLine_file import curLine, other_tag

all_entity_dict = defaultdict(dict)
before2after = {"父亲":"父亲", "赵磊":"赵磊", "甜蜜":"甜蜜", "大王叫我来":"大王叫我来巡山"}
def get_slot(param):
    slot = []
    if "<" not in param:
        return slot
    if ">" not in param:
        print(curLine(), "param:", param)
        return slot
    if "</" not in param:
        return slot
    start_segment = re.findall("<[\w_]*>", param)
    end_segment = re.findall("</[\w_]*>", param)
    if len(start_segment) != len(end_segment):
        print(curLine(), "start_segment:", start_segment)
        print(curLine(), "end_segment:", end_segment)
    search_location = 0
    for s,e in zip(start_segment, end_segment):
        entityType = s[1:-1]
        assert "</%s>" % entityType == e
        start_index = param[search_location:].index(s) + len(s)
        end_index = param[search_location:].index(e)
        entity_info = param[search_location:][start_index:end_index]
        search_location += end_index + len(e)
        before,after = entity_info, entity_info
        if "||" in entity_info:
            before, after = entity_info.split("||")
        if before in before2after:
            after = before2after[before]
        if before not in all_entity_dict[entityType]:
            all_entity_dict[entityType][before] = [after, 1]
        else:
            if after != all_entity_dict[entityType][before][0]:
                print(curLine(), entityType, before, after, all_entity_dict[entityType][before])
            assert after == all_entity_dict[entityType][before][0]
            all_entity_dict[entityType][before][1] += 1
        if before != after:
            before = after
            if before not in all_entity_dict[entityType]:
                all_entity_dict[entityType][before] = [after, 1]
            else:
                assert after == all_entity_dict[entityType][before][0]
                all_entity_dict[entityType][before][1] += 1


# 预处理
def process(source_file, train_file, dev_file):
    dev_lines = []
    train_num = 0
    intent_distribution = defaultdict(dict)
    with open(source_file, "r") as f, open(train_file, "w") as f_train:
        reader = csv.reader(f)
        train_write = csv.writer(f_train, dialect='excel')
        for row_id, line in enumerate(reader):
            if row_id==0:
                continue
            (sessionId, raw_query, domain_intent, param) = line
            get_slot(param)

            if domain_intent == other_tag:
                domain = other_tag
                intent = other_tag
            else:
                domain, intent = domain_intent.split(".")
            if intent in intent_distribution[domain]:
                intent_distribution[domain][intent] += 1
            else:
                intent_distribution[domain][intent] = 0
            if row_id == 0:
                continue
            sessionId = int(sessionId)
            if sessionId % 10>0:
                train_write.writerow(line)
                train_num += 1
            else:
                dev_lines.append(line)
    with open(dev_file, "w") as f_dev:
        write = csv.writer(f_dev, dialect='excel')
        for line in dev_lines:
            write.writerow(line)
    print(curLine(), "dev=%d, train=%d" % (len(dev_lines), train_num))
    for domain, intent_num in intent_distribution.items():
        print(curLine(), domain, intent_num)


if __name__ == "__main__":
    corpus_folder = "/home/wzk/Mywork/corpus/compe/69"
    source_file = os.path.join(corpus_folder, "train.csv")
    train_file = os.path.join(corpus_folder, "train.txt")
    dev_file = os.path.join(corpus_folder, "dev.txt")
    process(source_file, train_file, dev_file)

    for entityType, entityDict in all_entity_dict.items():
        json_file = os.path.join(corpus_folder, "%s.json" % entityType)
        with open(json_file, "w") as f:
            json.dump(entityDict, f, ensure_ascii=False, indent=4)
        print(curLine(), "save %d %s to %s" % (len(entityDict), entityType, json_file))

