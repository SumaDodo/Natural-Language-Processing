import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import Phrases
import logging
import re
import pandas as pd
import nltk.data
from openpyxl.formula import tokenizer
import re
import collections
import csv
import fileinput
from collections import Counter
import numpy as np
from itertools import tee, islice
from pprint import pprint
import os
import math

crs = open("gene-trainF17.txt", "r")  # read file
lists = []  # list to store the words
list_tag = []  # list of tags
unique_tag = []  # unique set of tags
unique_words = []  # unique set of words
word_tag = []
only_tag = []
pos_tag = []

for element in crs:
    words = element.strip().split()  # for every element in the file split and read
    if words:  # Skip Spaces/Empty lines
        if words[1] not in lists:
            lists.append(words[1])
        if words[2] not in list_tag:
            list_tag.append(words[2])  # copy tags to the list_tag
        word_tag.append(words[1])

crs2 = open("gene-trainF17.txt", "r")
for ele in crs2:
    words1 = ele.strip().split()  # for every element in the file split and read
    if words1:  # Skip Spaces/Empty lines
        only_tag.append(words1[2])

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#setting parameters
num_features = 300
min_word_count = 1
num_workers = 4
context = 6
downsampling = 1e-3
#vector representation for words
model = word2vec.Word2Vec([lists], workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "gene_oped"
model.save(model_name)
new_model = gensim.models.Word2Vec.load('gene_oped')

vocab = list(model.wv.vocab.keys())

#Vector representation for tags
model1 = word2vec.Word2Vec([list_tag], workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model1.init_sims(replace=True)

model1_name = "gene_oped1"
model1.save(model1_name)
new_model1 = gensim.models.Word2Vec.load('gene_oped1')

vocab1 = list(model1.wv.vocab.keys())

unique_tag = list_tag  # list(set(list_tag))
unique_words = lists  # list(set(lists))
size_tag = len(unique_tag)
size_words = len(unique_words)

word_counts = Counter(lists)  # count and order the words in the list
tag_counts = Counter(list_tag)  # count and order the tags

vocabulary = []
vocabulary = word_counts

# bigram counts
def bigram(list, gram):
    tag_bigram = list
    while True:
        one, two = tee(tag_bigram)
        length = tuple(islice(one, gram))
        if len(length) == gram:
            yield length
            next(two)
            tag_bigram = two
        else:
            break


bigram_list = Counter(bigram(only_tag, 4))

# using numpy to create 2D array for transition matrix
#  #matrix of type float and initialize to one - Laplace smoothing
trans_array = np.zeros((size_tag, size_tag), dtype=np.float64)
tag_array = np.array((3, 3), dtype=np.chararray)
tag_array = tag_counts

b1 = []
b2 = []
count = []
# transition matrix
for i, j in bigram_list.items():
    b1.append(i[0])
    b2.append(i[1])
    count.append(j)
    trans_array[unique_tag.index(i[0])][unique_tag.index(i[1])] = j

trans_div = np.divide(trans_array, np.sum(trans_array, axis=1)) + 1

emission_matrix = np.zeros((size_tag, size_words), dtype=np.float64)
crs1 = open("gene-trainF17.txt", "r")  # read file

for elements in crs1:
    words_tag = elements.strip().split()
    if words_tag:
        if (words_tag[2] in unique_tag) and (words_tag[1] in unique_words):
            emission_matrix[unique_tag.index(words_tag[2])][unique_words.index(words_tag[1])] += 1+((math.tanh(np.mean(model1[words_tag[2]],axis=0)))+(math.tanh(np.mean(model[words_tag[1]], axis=0))))

div = np.divide(emission_matrix, np.sum(emission_matrix, axis=0))
# Start Probabilities for the states
start_prob_matrix = np.zeros((1, 3), dtype=np.float64)
for states in unique_tag:
    if states == "O":
        k = unique_tag.index(states)
        start_prob_matrix[0][k] = 0.03
    elif states == "B":
        k = unique_tag.index(states)
        start_prob_matrix[0][k] = 0.05
    else:
        k = unique_tag.index(states)
        start_prob_matrix[0][k] = 0.02

# Viterbi algorithm implementation
observation_words = []
obs = open("F17-assgn4-test.txt", "r")

for letters in obs:
    test = letters.strip().split()
    if test:
        observation_words.append(test[1])

viterbi_matix = np.zeros((((len(unique_tag))), (len(observation_words))), dtype=np.float64)
backpointer_matrix = np.zeros(((len(unique_tag)), len(observation_words)), dtype=np.int64)

for observation in unique_tag:
    viterbi_matix[unique_tag.index(observation)][0] = (
    start_prob_matrix[0][unique_tag.index((observation))] * div[unique_tag.index(observation)][
        unique_words.index(observation_words[0])])

for each_words in range(1, len(observation_words)):
    for state in unique_tag:
        max_val = 1
        for previous_state in unique_tag:
            k = viterbi_matix[unique_tag.index(previous_state)][each_words - 1] * \
                trans_div[unique_tag.index(previous_state)][unique_tag.index(state)]
            if (k > max_val):
                max_val = k
                backpointer_matrix[unique_tag.index(state)][each_words] = unique_tag.index(previous_state)
                if(backpointer_matrix[unique_tag.index(state)][each_words] == 2 and backpointer_matrix[unique_tag.index(state)][each_words-1]== 0):
                    backpointer_matrix[unique_tag.index(state)][each_words-1] = 2
                if (observation_words[each_words] == '-' and backpointer_matrix[unique_tag.index(state)][each_words-1] == 1):
                    backpointer_matrix[unique_tag.index(state)][each_words] = 2
                if (backpointer_matrix[unique_tag.index(state)][each_words-1] == 0 and backpointer_matrix[unique_tag.index(state)][each_words]== 2):
                    backpointer_matrix[unique_tag.index(state)][each_words] = 1 #correcting the words recognised as I to B
        if observation_words[each_words] in unique_words:
            if (re.match('[A-Z]+$', observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] =max_val * div[unique_tag.index(state)][unique_words.index(observation_words[each_words])]
                val = viterbi_matix[unique_tag.index(state)][each_words]
            elif (re.match('\d+[a-zA-Z]+$', observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div[unique_tag.index(state)][unique_words.index(observation_words[each_words])]
                val1 = viterbi_matix[unique_tag.index(state)][each_words]
            elif (re.match('[A-Za-z]+$', observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div[unique_tag.index(state)][
                    unique_words.index(observation_words[each_words])]
                val2 = viterbi_matix[unique_tag.index(state)][each_words]
            else:
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div[unique_tag.index(state)][unique_words.index(observation_words[each_words])]
        else:#matching the shape of unknown words
            if not (re.search('\w', observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.max()
            elif (re.match('-',observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.min()
            elif (re.match('[A-Z]+$', observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.max()#val
            elif (re.match('[A-Za-z]+$', observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.max()
            elif (re.search('\d+[a-zA-Z]+$',observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.max()#val
            elif(re.match('((?:[a-z][a-z]*[0-9]+[a-z0-9]*))',observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.max()#val
            elif re.match('\w+bulin\b',observation_words[each_words]):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div[unique_tag.index(state)][unique_words.index(re.match('\w+bulin\b',observation_words[each_words]))]#* div.min()
            elif(re.match('[a-z]+$',observation_words[each_words])):
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.min()
            else:
                viterbi_matix[unique_tag.index(state)][each_words] = max_val * div.min()

test_tags = []
i = 0
# unique_tag
test_tags.append(np.argmax(viterbi_matix[:, len(observation_words) - 1]))
for i in range(len(observation_words) - 1, 0, -1):
    test_tags.append(backpointer_matrix[test_tags[-1]][i])
final = []
final = (list(reversed((test_tags))))
i = 0
while i < len(observation_words):
    # print(unique_tag[final[i]])
    i += 1

obs.close()
crs1.close()

obs = open("F17-assgn4-test.txt", "r")
output = open("Dodmani-Suma-assgn4-out.txt", "w")
i = 0
for l1 in obs:
    test = l1.strip().split()
    if test:
        # if (test[1] == '-' and unique_tag[final[i-1]] == 'B'):
        #     # unique_tag[final[i]] = 'I'
        #     # print(unique_tag[final[i]])
        #     output.write(test[0] + "\t" + test[1] + "\t" + 'I' + "\n")
        #     print("after"+ unique_tag[final[i]])
        # elif(test[1]-1 == '-' and unique_tag[final[i-1]] == 'I'):
        #     output.write(test[0] + "\t" + test[1] + "\t" + 'I' + "\n")
        # else:
        output.write(test[0] + "\t" + test[1] + "\t" + unique_tag[final[i]] + "\n")
        i = i + 1
        #if (test[1] == "."):
        #if (test[1] == "\n"):
            #output.write("\n")
    else:
        output.write("\n")

obs.close()
output.close()
