from __future__ import unicode_literals
from hazm import *
from efficient_apriori import apriori
from prettytable import PrettyTable
from scipy import stats


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
    for sentence in tags:
        print(sentence)


def cleaner(tags):
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    uniq_tags_list = []
    token_set = set()
    for sent in tags:
        tmp_list = []

        for token in sent:
            tmp_list.append(list(token))
            for item in tmp_list:

                if item[1] == 'N':
                    item[0] = stemmer.stem(item[0])
                if item[1] == 'V':
                    item[0] = lemmatizer.lemmatize(item[0])
                token_set.add(item[0])

        uniq_tags_list.append(tmp_list)

    uniq_token = []
    for sentence in uniq_tags_list:
        for words in sentence:
            uniq_token.append(words[0])
    return uniq_token, uniq_tags_list


def show_apriori_table(min_support):
    table = PrettyTable()
    table.field_names = ['lift', 'confidence', 'support', 'word']

    item, rules = apriori(sentence, min_support=min_support)
    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
        table.add_row([rule.lhs, rule.support, rule.confidence, rule.lift])


def show_chi_square(uniq_token, uniq_tags_list):
    table = []
    for word in uniq_token:
        freq_list = []
        for sentence in uniq_tags_list:
            frequncy_in_sentence = 0
            for words in sentence:
                if word == words[0]:
                    frequncy_in_sentence += 1

            freq_list.append(frequncy_in_sentence)
        table.append(freq_list)

    chi2_stat, p_val, dof, ex = stats.chi2_contingency(table)
    print('chi_square:  ', chi2_stat)
    print('p_value:  ', p_val)
    print('degree of freedome:  ', dof)


sentence, tags = divide_by_sentence()
show_tags(tags)
uniq_token, uniq_tags_list = cleaner(tags)
show_chi_square(uniq_token, uniq_tags_list)
