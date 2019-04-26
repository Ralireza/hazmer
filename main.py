from __future__ import unicode_literals
from hazm import *
from efficient_apriori import apriori
from prettytable import PrettyTable

stemmer = Stemmer()
lemmatizer = Lemmatizer()


def divide_by_sentence():
    f = open("n.txt", "r")
    tagger = POSTagger(model='resources/postagger.model')
    sentence_list = []
    sentence_tag_list = []
    for line in f.readlines():
        normalizer = Normalizer()
        normalized = normalizer.normalize(line)
        word_token = word_tokenize(normalized)
        word_tag = tagger.tag(word_token)
        split_index = 0
        # fold sentence by ariving to V tag
        for i in range(len(word_tag)):
            if word_tag[i][1] == 'V':
                sentence_list.append(word_token[split_index:(i + 1)])
                sentence_tag_list.append(word_tag[split_index:(i + 1)])
                split_index = i + 1
    f.close()
    return sentence_list, sentence_tag_list


def show_tags(tags):
    print(tags)


sentence, tags = divide_by_sentence()

tags_list = []
token_set = set()

for sent in tags:
    tmp_list = []

    for token in sent:
        tmp_list.append(list(token))
        for item in tmp_list:

            # if item[1]=='N':
            #     item[0]=stemmer.stem(item[0])
            if item[1] == 'V':
                item[0] = lemmatizer.lemmatize(item[0])
            token_set.add(item[0])

    tags_list.append(tmp_list)


def show_apriori_table():
    table = PrettyTable()
    table.field_names = ['conviction', 'lift', 'confidence', 'support', 'word']

    item, rules = apriori(sentence, min_support=0.01)
    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
        table.add_row([rule.lhs, rule.support, rule.confidence, rule.lift, rule.conviction])

    print(table)

show_apriori_table()