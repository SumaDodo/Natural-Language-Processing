import re
import collections
import csv
import fileinput
from collections import Counter
import numpy as np
from itertools import tee,islice
from pprint import pprint
import os
import math
import csv, string, nltk, pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import  nltk

count_neg = 0
count_pos = 0
adv_count = 0
word_neg = []
word_pos = []
file_neg = open("hotelF-train.txt", encoding="utf8")
file_pos = open("hotelT-train.txt",encoding="utf8")

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

#Taking document priors
for elements in file_neg:
    for line in file_neg:
        count_neg += 1

for element1 in file_pos:
    for lines in file_pos:
        count_pos += 1

total_reviews = count_pos+count_neg
doc_prior_pos = (count_pos/ total_reviews)
doc_prior_neg = (count_neg/ total_reviews)

review_pos_count = {}
review_neg_count = {}
test_data = {}
all_reviews = {}
all_reviews_count = {}
total_false_verb_count = 0
total_true_verb_count = 0
total_false_adj_count = 0
total_true_adj_count =0
total_false_adv_count = 1
total_true_adv_count =1
punct_count_true = 0
punct_count_false = 0

#Count of each word in the negative dataset with pre-processing
crs = open ("hotelF-train.txt", encoding= "utf8")

for reviews in crs:
    read_word = reviews.lower().split()[1:]
    for each_word in read_word:
        if (each_word not in stop_words):#Removing stop words
            if (re.sub("[-!,?.;()@#$]","",each_word)):
                punct_count_false = punct_count_false+1
            each_word = re.sub("[-!,?.;()@#$]","", each_word)
            each_word = wordnet_lemmatizer.lemmatize(each_word)#Lemmatize
            if (each_word not in review_neg_count):
                review_neg_count[each_word] = 1
            else:
                review_neg_count[each_word]+= 1
            if (each_word not in '-'):
                tagger = nltk.pos_tag([each_word])
                if (tagger[0][1] in ('PRP','CC','NN','NNS')):#Check for nouns and pronouns
                    total_false_verb_count = total_false_verb_count +1
                elif (tagger[0][1] in ('RB','JJS','PRP$','JJ','VB','VBZ','VBD','VBN','VBG')):#Check for verbs and adjective
                    total_false_adj_count = total_false_adj_count + 1
            if each_word in ('adjoining','rooms','amenities','baggage','bed','breakfast','book','booked','checkin','checkout','complimentary','deposit','floor','guests','reservation'):
                total_false_adv_count = total_false_adv_count +1


c =0
for i1,j1 in review_neg_count.items():
    c = c + j1
#Count of ech word in the positive dataset with preprocessing
crs1 = open ("hotelT-train.txt", encoding= "utf8")

for reviews1 in crs1:
    read_pos = reviews1.lower().split()[1:]
    for each_pos in read_pos:
        if each_pos not in stop_words:#stop word removal
            if (re.sub("[-!,?.;()@#$]","",each_pos)):
                punct_count_true = punct_count_true + 1
            each_pos = re.sub("[!,?.;()@#$-]","", each_pos)
            each_pos = wordnet_lemmatizer.lemmatize(each_pos)#lemmatization
            if each_pos not in review_pos_count:
                review_pos_count[each_pos] = 1
            else:
                review_pos_count[each_pos]+= 1
            if (each_pos not in '-'):
                tagger = nltk.pos_tag([each_pos])
                if (tagger[0][1] in ('PRP','CC','NN','NNS')):#count of nouns and pronoun
                    total_true_verb_count = total_true_verb_count + 1
                elif (tagger[0][1] in ('RB','JJS','PRP$','JJ','VB','VBZ','VBD','VBN','VBG',)):#check for verb and adjective
                    total_true_adj_count = total_true_adj_count + 1
            if each_word in ('adjoining','rooms','amenities','baggage','bed','breakfast','book','booked','checkin','checkout','complimentary','deposit','floor','guests','reservation'):
                total_true_adv_count = total_true_adv_count +1

pos = 0
for i2,j2 in review_pos_count.items():
    pos = pos + j2
#Combining positive and neagtive
all_reviews.update(review_pos_count)
all_reviews.update(review_neg_count)
v1 = len(all_reviews)

#Calculating the probability of each word belonging to a particular class
probability_pos = {}
probability_neg = {}

for k, v in review_neg_count.items():
    probability_neg[k] = ((v+1)/(c+v1))

for k1, vl in review_pos_count.items():
    probability_pos[k1] = (1 + vl)/(pos+v1)

for k2,v2 in all_reviews.items():
    if (k2 not in probability_neg):
        probability_neg[k2] = 1 /(c+v1)
    if (k2 not in probability_pos):
        probability_pos[k2] = 1/(pos+v1)

total_false_verb_count = total_false_verb_count/(c+v1)
total_true_verb_count = total_true_verb_count/(pos+v1)
total_false_adj_count = total_false_adj_count/(c+v1)
total_true_adj_count = total_true_adj_count/(pos+v1)
total_false_adv_count = total_false_adv_count/(c+v1)
total_true_adv_count = total_true_adv_count/(pos+v1)
punct_count_false = punct_count_false/(pos+v1)
punct_count_true = punct_count_true /(c+v1)

#Test Data
#Taking out the positive and negative probabilities of each word in the sentence
word_neg_value = {}
word_pos_value = {}
test = open("hotelDeceptionTest.txt",encoding="utf8")
output = open("dodmani-suma-extra-out.txt","w")
w3 = test.readline()
count_adj = 0
count_adv = 0
count_verb = []
pos_weight = 0
neg_weight = 0
while w3:
    count_verb = []
    count_pronoun = 0
    count_pronoun1 = 0
    count = 0
    countn = 0
    count_pos = 0
    count_neg = 0
    count_advp = 0
    count_advn = 0
    punct_count_pos = 0
    punct_count_neg = 0
    rr = w3.lower().split()[1:]
    word_neg_value = {}
    word_pos_value = {}
    for w1 in rr:
        if w1 not in stop_words:
            if (re.sub("[!,?.;()@#$-]","",w1)):
                punct_count_pos = punct_count_pos + 1
                punct_count_neg = punct_count_neg + 1
            w1 = re.sub("[!,?.;()@#$-]","", w1)
            w1 = wordnet_lemmatizer.lemmatize(w1)
            if w1 in ('adjoining','rooms','amenities','baggage','bed','breakfast','book','booked','checkin','checkout','complimentary','deposit','floor','guests','reservation'):
                count_advp = count_advp +1
                count_advn = count_advn + 1
            if (w1 not in '-'):
                tagger = nltk.pos_tag([w1])
                if (tagger[0][1] in ('RB','JJS','PRP$','JJ','VB','VBZ','VBD','VBN','VBG')):
                    count = count + 1
                    countn = countn + 1
                    count_verb.append(tagger[0][1])
                if (tagger[0][1] in ('PRP','CC','NN','NNS')):
                    count_pos = count_pos +1
                    count_neg = count_neg + 1
            if (w1 in ('it','they','them','their')):
                count_pronoun = count_pronoun + 1
            if (w1 in ('I','me','we','myself','he','she')):
                count_pronoun1 = count_pronoun1 + 1
            if w1 in all_reviews:
                word_pos_value[w1] = probability_pos[w1]
            if w1 in all_reviews:
                word_neg_value[w1] = probability_neg[w1]
    value_pos = 1
    value_neg = 1
    count = count/(total_true_adj_count)
    countn = countn/(total_false_adj_count)
    count_pos = count_pos/(total_true_verb_count)
    count_neg = count_neg/(total_false_verb_count)
    punct_count_pos = punct_count_pos / ( punct_count_true)
    punct_count_neg = punct_count_neg / (punct_count_false)
    count_advp = count_advp/total_true_adv_count
    count_advn = count_advn/total_false_adv_count
    for h1,h2 in word_pos_value.items():
        value_pos = value_pos * h2
    value_pos = value_pos * doc_prior_pos * count * count_pos * punct_count_pos * count_advp
    for h3,h4 in word_neg_value.items():
        value_neg = value_neg * h4
    value_neg = value_neg * doc_prior_neg * countn * count_neg * punct_count_neg * count_advn
    if ((value_pos > value_neg)):
        output.write(w3[0]+w3[1]+w3[2]+w3[3]+w3[4]+w3[5]+w3[6]+"\t"+"T"+"\n")
    else:
        output.write(w3[0] + w3[1] + w3[2] + w3[3] + w3[4] + w3[5] + w3[6]+ "\t" + "F"+"\n")
    w3 = test.readline()
output.close()
