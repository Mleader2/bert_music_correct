# -*- coding: utf-8 -*-
# 混淆词提取算法
from pypinyin import lazy_pinyin, pinyin, Style
from tqdm import tqdm
import json
from Levenshtein import distance # python-Levenshtein
# 编辑距离 (Levenshtein Distance算法)
import os
import jieba
import re
import dimsim
import time


with open('./data/wordlists.json', 'r', encoding='utf8') as fi:
    word_list = json.load(fi)

with open('./corpus/1000.txt', 'r', encoding='utf8') as reader:
    cc_table = reader.read()

two_chars = [a for a in word_list if len(a) == 2]
three_chars = [a for a in word_list if len(a) == 3]
four_chars = [a for a in word_list if len(a) == 4]
print(two_chars)
print(three_chars)
print(four_chars)


class Confusion_set:
    def __init__(self):
        if os.path.exists('./data/confusion_set.json') and os.path.exists('./data/whitelist.json'):
            self.load()
        else:
            self.confusion_set = dict()
            self.whitelist = set()

    def load(self):
        print('\n-----loading start-----\n')
        with open('./data/confusion_set.json', 'r', encoding='utf8') as cs:
            self.confusion_set = json.load(cs)
        with open('./data/whitelist.json', 'r', encoding='utf8') as wl:
            self.whitelist = set(json.load(wl))

        all_addings = {}
        all_delets = {}
        old_cs = set(self.confusion_set)
        new_cs = set()
        for word in word_list:
            _new_cs = set()
            addings = set()
            if os.path.exists('./confusion_set/'+word+'.txt'):
                with open('./confusion_set/'+word+'.txt', 'r', encoding='utf8') as fi:
                    for line in fi:
                        new_cs.add(line.rstrip('\n'))
                        _new_cs.add(line.rstrip('\n'))
                addings = _new_cs -old_cs
                if len(addings):
                    all_addings[word] = addings
                for i in addings:
                    self.confusion_set[i] = word

        delets = old_cs - new_cs
        for i in delets:
            if not self.confusion_set[i] in all_delets:
                all_delets[self.confusion_set[i]] = {i}
            else:
                all_delets[self.confusion_set[i]].add(i)
            self.confusion_set.pop(i)
            self.whitelist.add(i)
        if len(all_addings):
            print('new addings:','\n')
            for i in all_addings:
                print('\t',i,': ',' '.join(list(all_addings[i])))
        else:
            print('no new addings')
        print()
        if len(all_delets):
            print('delets:','\n')
            for i in all_delets:
                print('\t',i,': ',' '.join(list(all_delets[i])))
        else:
            print('no delets')

        
    def dump(self):
        with open('./data/confusion_set.json', 'w', encoding='utf-8') as cs:
            json.dump(self.confusion_set, cs, ensure_ascii=False)
        with open('./data/whitelist.json', 'w', encoding='utf-8') as wl:
            json.dump(list(self.whitelist), wl, ensure_ascii=False)

        if not os.path.exists('./confusion_set'):
            os.makedirs('./confusion_set')

        for word in word_list:
            with open('./confusion_set/'+word+'.txt', 'w', encoding='utf8') as fo:
                for key, value in self.confusion_set.items():
                    if value == word:
                        fo.write(key+'\n')

    def confusion_by_dimsim(self): # 词少　直接得到相似度
        for word in word_list:
            candidates = []
            count_error = 0
            try:
                candidates = dimsim.get_candidates(word, mode='simplified', theta=1)
                for candidate in candidates:
                    self.confusion_set[candidate] = word
            except Exception as e:
                count_error += 1
                continue

        print('confusion_set', self.confusion_set)
        print('count_error', count_error)

        with open('./data/confusion_set.json', 'w', encoding='utf8') as cs:
            json.dump(self.confusion_set, cs, ensure_ascii=False)
        return self.confusion_set

    def confusion_by_cons_vow(self):
        vowe_table = {'ai':'ei', 'an':'ang', 'en':'eng', 'in':'ing',
                      'ei':'ai', 'ang':'an', 'eng':'en', 'ing':'in'}

        for highf_words in word_list:
            length = len(highf_words)
            if length == 2: #长度为2的高频词
                word1 = highf_words[0]
                word2 = highf_words[1]
                pyin_cons = pinyin(word1, style=Style.INITIALS)


            elif length == 3:
                pass
            elif length == 4:
                pass

    def confusion_cons(self, word):
        cons_table = {'n':'l', 'l':'n', 'f':'h', 'h':'f', 'zh':'z', 'z':'zh', 'c':'ch', 'ch':'c', 's':'sh', 'sh':'s'}
        vowe_table = {'ai': 'ei', 'an': 'ang', 'en': 'eng', 'in': 'ing',
                      'ei': 'ai', 'ang': 'an', 'eng': 'en', 'ing': 'in'}

        word_pyin = lazy_pinyin(word)[0]
        pyin_cons = pinyin(word, style=Style.INITIALS)
        pyin_vowe = pinyin(word, style=Style.FINALS)

        cons_dict = {}
        if pyin_cons in cons_table:
            word_pyin.replace(pyin_cons, cons_table[pyin_cons])
            for confu_word in cc_table:
                confu_word_pyin = lazy_pinyin(confu_word)[0]
                if word_pyin == confu_word_pyin:
                    cons_dict[confu_word] = word

        pass

    def confusion_vow(self):
        pass

    def create_highword_confu(self):

        highf_word_confu = {}
        for highf_words in word_list:
            for highf_word in highf_words:
                for character in cc_table:
                    highf_word_pyin = lazy_pinyin(highf_word)[0]
                    character_pyin = lazy_pinyin(character)[0]
                    confu_highf = highf_word_confu.get(highf_word, [])
                    edit_dis = distance(highf_word_pyin, character_pyin)
                    if edit_dis == 1:
                        confu_highf.append(character)
                        highf_word_confu[highf_word] = confu_highf #该dict, key为高频词中的字，value为1000字典中编辑距离等于一的混淆字

        print('highf_word_confu', (len(highf_word_confu)))
        with open('./data/highf_word_confu.json', 'w', encoding='utf8') as writer:
            json.dump(highf_word_confu, writer, ensure_ascii=False)
        return highf_word_confu

    def confusion_by_cctable(self):
        highf_word_confu = self.create_highword_confu()
        highf_phrase_confu = {}
        for chars in two_chars:
            char1 = chars[0]
            char2 = chars[1]
            for confu in highf_word_confu[char1]:
                phrase_confu = confu+char2
                highf_phrase_confu[phrase_confu] = chars
            for confu in highf_word_confu[char2]:
                phrase_confu = char1+confu
                highf_phrase_confu[phrase_confu] = chars

        for chars in three_chars:
            char1 = chars[0]
            char2 = chars[1]
            char3 = chars[2]
            for confu in highf_word_confu[char1]:
                phrase_confu = confu+char2+char3
                highf_phrase_confu[phrase_confu] = chars
            for confu in highf_word_confu[char2]:
                phrase_confu = char1+confu+char3
                highf_phrase_confu[phrase_confu] = chars
            for confu in highf_word_confu[char3]:
                phrase_confu = char1+char2+confu
                highf_phrase_confu[phrase_confu] = chars

        for chars in four_chars:
            char1 = chars[0]
            char2 = chars[1]
            char3 = chars[2]
            char4 = chars[3]
            for confu in highf_word_confu[char1]:
                phrase_confu = confu+char2+char3+char4
                highf_phrase_confu[phrase_confu] = chars
            for confu in highf_word_confu[char2]:
                phrase_confu = char1+confu+char3+char4
                highf_phrase_confu[phrase_confu] = chars
            for confu in highf_word_confu[char3]:
                phrase_confu = char1+char2+confu+char4
                highf_phrase_confu[phrase_confu] = chars
            for confu in highf_word_confu[char4]:
                phrase_confu = char1+char2+char3+confu
                highf_phrase_confu[phrase_confu] = chars

            print('highf_phrase_confu', (len(highf_phrase_confu)))
            self.confusion_set = highf_phrase_confu
            print('self.confusion_set', self.confusion_set)
            with open('./data/highf_phrase_confu.json', 'w', encoding='utf8') as writer:
                json.dump(highf_phrase_confu, writer, ensure_ascii=False)
            return highf_phrase_confu

    def create_confusion_set(self):
        with open('./test_jindi/log_jindi.json', 'r', encoding='utf-8') as fi:
            lines = json.load(fi)
        lines_N = set()
        for line in lines:
            _ = set([obj for obj in re.sub('[0-9]+', 'N',
                        re.sub('[《》]+', ' ', re.sub('[^a-zA-Z0-9\u4e00-\u9fa5《》]+', ' ', line))
                        ).split(' ') if len(obj) > 2])
            for _2 in _:
                lines_N.add(_2)

        original_sentence = dict()
        for line in lines_N:
            print(line)
            cut_line = jieba.lcut(line)
            i = 0
            cut_idx = [i]
            for word in cut_line:
                i += len(word)
                cut_idx.append(i)
            pinyin = lazy_pinyin(line)
            n = len(pinyin)
            for i in range(n-1):
                for word in two_chars:
                    word_pinyin = ''.join(lazy_pinyin(word))
                    #编辑距离小于2，且不为原字，没有包含其他词语（如 “洛杉矶机场” 不能被切为 “洛杉|矶机场（进机场）”)，不在白名单中
                    if distance(pinyin[i] + pinyin[i+1], word_pinyin) < 2 and not line[i:i+2] == word and i in cut_idx and i+2 in cut_idx and not line[i:i+2] in self.whitelist:
                        self.confusion_set[line[i:i+2]] = word
                        original_sentence[line[i:i+2]] = line
                if i < n-2:
                    for word in three_chars:
                        word_pinyin = ''.join(lazy_pinyin(word))
                        if distance(pinyin[i]+pinyin[i+1]+pinyin[i+2], word_pinyin) < 2 and not line[i:i+3] == word and i in cut_idx and i+3 in cut_idx and not line[i:i+3] in self.whitelist:
                            self.confusion_set[line[i:i+3]] = word
                            original_sentence[line[i:i+3]] = line
                if i < n-3:
                    for word in four_chars:
                        word_pinyin = ''.join(lazy_pinyin(word))
                        if distance(pinyin[i]+pinyin[i+1]+pinyin[i+2]+pinyin[i+3],word_pinyin) <4 and not line[i:i+4] == word and i in cut_idx and i+4 in cut_idx and not line[i:i+4] in self.whitelist:
                            self.confusion_set[line[i:i+4]] = word
                            original_sentence[line[i:i+4]] = line
        print('confusion_set', self.confusion_set)

        with open('./data/oringinal_sentence.json', 'w', encoding='utf-8') as fo:
            json.dump(original_sentence, fo, ensure_ascii=False)

        with open('./data/confusion_set.json', 'w', encoding='utf8') as cs:
            json.dump(self.confusion_set, cs, ensure_ascii=False)
        return self.confusion_set, original_sentence

    def add_dict(self):
        fi = open ('./data/new_online.txt','r',encoding='utf-8')
        lines = [line for line in fi]

        original_sentence = dict()
        for line in tqdm(lines):

            cut_line = jieba.lcut(line)
            i= 0
            cut_point = [0]
            for word in cut_line:
                i += len(word)
                cut_point.append(i)
            pinyin = lazy_pinyin(line)
            for i in range(len(pinyin)-2):
                for word, word_pinyin in two_chars:
                    #编辑距离小于2，且不为原字，没有包含其他词语（如 “洛杉矶机场” 不能被切为 “洛杉|矶机场（进机场）”)，不在白名单中
                    if distance(pinyin[i] + pinyin[i+1], word_pinyin) < 2 and not line[i:i+2] == word and i in cut_point and i+2 in cut_point and not line[i:i+2] in self.whitelist:
                        self.confusion_set[line[i:i+2]] = word
                        original_sentence[line[i:i+2]] = line
                if i < len(pinyin)-2:
                    for word,word_pinyin in three_chars:        
                        if distance(pinyin[i]+pinyin[i+1]+pinyin[i+2],word_pinyin) <2 and not line[i:i+3] == word and i in cut_point and i+3 in cut_point and not line[i:i+3] in self.whitelist:
                            self.confusion_set[line[i:i+3]] = word
                            original_sentence[line[i:i+3]] = line
                if i < len(pinyin)-3:
                    for word,word_pinyin in four_chars:
                        if distance(pinyin[i]+pinyin[i+1]+pinyin[i+2]+pinyin[i+3],word_pinyin) <3 and not line[i:i+4] == word and i in cut_point and i+4 in cut_point and not line[i:i+4] in self.whitelist:
                            self.confusion_set[line[i:i+4]] = word
                            original_sentence[line[i:i+4]] = line
        with open('oringinal_sentence.json', 'w') as fo:
            json.dump(original_sentence, fo, ensure_ascii=False)

        with open('confusion_set.json', 'w') as cs:
            json.dump(self.confusion_set, cs, ensure_ascii=False)

    def search(self,word):
        with open('original_sentence.json','r') as fi:
            original_sentence = json.load(fi)
        return(original_sentence[word])
    def unblock(self):
        print('confusion set(before): ', len(self.confusion_set))
        print('white list(before): ', len(self.whitelist))
        with open('show.txt','r',encoding='utf8') as fi:
            for line in fi:
                line = line.split(' ')
                if len(line) == 3:
                    self.confusion_set.pop(line[0])
                    self.whitelist.add(line[0])
        print('confusion set(after): ', len(self.confusion_set))
        print('white list(after): ', len(self.whitelist))

    def judge(self,sentence):
        for it in range(len(sentence)):
            if(it <= len(sentence)-4) and sentence[it:it+4] in self.confusion_set:
                sentence = sentence[:it]+self.confusion_set[sentence[it:it+4]]+sentence[it+4:]

            if(it <= len(sentence)-3) and sentence[it:it+3] in self.confusion_set:
                sentence = sentence[:it]+self.confusion_set[sentence[it:it+3]]+sentence[it+3:]

            if(it <= len(sentence)-2) and sentence[it:it+2] in self.confusion_set:
                sentence = sentence[:it]+self.confusion_set[sentence[it:it+2]]+sentence[it+2:]

        return sentence


if __name__ == "__main__":
    a = Confusion_set()
    a.create_confusion_set()
    #a.confusion_by_cctable()
    #a.confusion_by_dimsim()
    print(len(a.confusion_set))
    #a.dump()
