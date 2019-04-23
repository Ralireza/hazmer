from __future__ import unicode_literals
import pandas as pd
from hazm import *
from apyori import apriori

# tagger = POSTagger(model='resources/postagger.model')
# s = tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم.ما بسیار کتاب می‌خوانیم.ما بسیار کتاب می‌خوانیم.ما بسیار کتاب می‌خوانیم'))
# print(s)

vocab = set()
doc_vocab = []
number_of_terms = 0
number_of_docs = 0
class_dictionary = {}
cls_index = 0
doc_clss_index = []
count_of_that_class = []
class_name = []
f = open("n.txt", "r")
# t=f.readline()
# fl =f.readlines()
# for x in fl:
#     print(x)

mylen = 0
for line in f.readlines():
    normalizer = Normalizer()
    t = normalizer.normalize(line)

    # print(sent_tokenize(line))
    # print(word_tokenize(line))
    mylen += len(word_tokenize(line))
    print(len(word_tokenize(line)))
    print(mylen)
f.close()
# h = open("n.txt", "w+")

# with open('HAM-Train-Test/HAM-Train.txt', 'r', encoding="utf8") as infile:
#     for line in infile:
#         number_of_docs += 1
#         cls, sep, text = line.partition('@@@@@@@@@@')
#         if number_of_docs < 20:
#             print(text)
#             h.write(text)
#
#         # assigning class index for each document
#         if (class_dictionary.get(cls)) is None:
#             class_dictionary[cls] = cls_index
#             tmp = cls_index
#             cls_index += 1
#             count_of_that_class.append(1)
#             class_name.append(cls)
#
#         else:
#             tmp = class_dictionary[cls]
#             count_of_that_class[tmp] += 1
#         doc_clss_index.append(tmp)
#         tokens = word_tokenize(text)
#         tmp_set = set()
#         number_of_terms += len(tokens)
#         for word in tokens:
#             vocab.add(word)
#             tmp_set.add(word)
#         doc_vocab.append(tmp_set)

# print("vocab size:", len(vocab))
# print("number of terms (all tokens):", number_of_terms)
# print("number of docs:", number_of_docs)
# print("number of classes:", cls_index)
