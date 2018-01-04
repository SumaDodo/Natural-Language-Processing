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

count_neg = 0
count_pos = 0
word_neg = []
word_pos = []
file_neg = open("hotelNegT-train.txt", encoding="utf8")
file_pos = open("hotelPosT-train.txt",encoding="utf8")

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


#Count of each word in the negative dataset with pre-processing
crs = open ("hotelNegT-train.txt", encoding= "utf8")

for reviews in crs:
    read_word = reviews.lower().split()[1:]
    for each_word in read_word:
        each_word = re.sub("[!@#$]","", each_word)
        if (each_word not in review_neg_count):
            review_neg_count[each_word] = 1
        else:
            review_neg_count[each_word]+= 1

c =0
for i1,j1 in review_neg_count.items():
    c = c + j1
#Count of ech word in the positive dataset with preprocessing
crs1 = open ("hotelPosT-train.txt", encoding= "utf8")

for reviews1 in crs1:
    read_pos = reviews1.lower().split()[1:]
    for each_pos in read_pos:
        each_pos = re.sub("[!@#$]","", each_pos)
        if each_pos not in review_pos_count:
            review_pos_count[each_pos] = 1
        else:
            review_pos_count[each_pos]+= 1

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

#Test Data
#Taking out the positive and negative probabilities of each word in the sentence
word_neg_value = {}
word_pos_value = {}
test = open("HW3-testset.txt",encoding="utf8")
output = open("dodmani-suma-assgn3-out.txt","w")
w3 = test.readline()
while w3:
    rr = w3.lower().split()[1:]
    word_neg_value = {}
    word_pos_value = {}
    for w1 in rr:
        w1 = re.sub("[!@#$]","", w1)
        if w1 in all_reviews:
            word_pos_value[w1] = probability_pos[w1]
        if w1 in all_reviews:
            word_neg_value[w1] = probability_neg[w1]
    value_pos = 1
    value_neg = 1
    for h1,h2 in word_pos_value.items():
        value_pos = value_pos * h2
    value_pos = value_pos * doc_prior_pos
    for h3,h4 in word_neg_value.items():
        value_neg = value_neg * h4
    value_neg = value_neg * doc_prior_neg
    if (value_pos > value_neg):
        output.write(w3[0]+w3[1]+w3[2]+w3[3]+w3[4]+w3[5]+w3[6]+"\t"+"POS"+"\n")
    else:
        output.write(w3[0] + w3[1] + w3[2] + w3[3] + w3[4] + w3[5] + w3[6]+ "\t" + "NEG"+"\n")
    w3 = test.readline()
output.close()
