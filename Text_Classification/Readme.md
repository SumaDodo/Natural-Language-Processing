**Understanding:**

When we try understanding the opinion of a writer about his or her view towards a topic, we
are doing an opining mining. This is very helpful in understanding the common opinions of
people in general towards a subject which in turn might be used further. This opinion mining
is referred to as sentiment analysis. Categorizing the opinions [mainly positive or negative] of
the writer on a particular topic. For that we classify the given data and categorize it.

**Design:**  
The system was trained on the training set given but during the training phase, the data
provided was divided as training and dev in 90-10 ratio and the system was trained based on
this. The system is expected to calculate the respective word probability of belonging to a
particular class such that when it is fed with the test data, for each sentence it is expected to
check the each word’s probability of belonging to a particular class and thus we get the total
inclination of the sentence to a particular class.

**Implementation:**  
  1] Firstly, the system was designed without taking out the stop words and also without any
pre-processing of the data. It was obvious that systems accuracy will be less, given that the
upper case and lower case characters will be identified distinctly even if the words are same.
And also, the words ending with punctuation and the same words without any punctuation
would also behave in the similar fashion.  
  2] The system was modified with data pre-processing steps included. And a very good
accuracy rate was observed.  
  3] The system was then tested with respect to the stop words if they would make any
difference to the accuracy. But as expected the accuracy after removing the stop words and
before removing the stop words didn’t bring much change to the system’s accuracy.  
  4] The basic model of Naïve Bayes for Sentiment Analysis is implemented here. No
additional libraries are used. For the training data, it is pre-processed and the probability of
each word occurring in respective class is calculated and if the word doesn’t belong to a
particular class, then Laplace smoothing is applied to handle such situation.  
  5] For the test data, the probability of each word in the sentence belonging to a particular
class is checked from the training and is multiplied with the prior probability of the class in
the dataset. Thus the sentiment if the review is either positive or negative is obtained.  

**Results and discussion:**  
In the initial phase, the training was divided 90-10. It was trained on 90% of data and tested
on the rest, the accuracy observed was 90%. This is due to the total independence of the
words from each other. On the test set, the system's accuracy observed was 88%

**Improvements:**  
The following are few improvements that can be made to the system:  
  1] Lemmatization is essential in this aspect, given the fact that it is common to observe words
like – like, liked, liking, likings etc. and without lemmatization, these words end up being
counted separately when in fact the meaning seems to be the same. Thus, lemmatization is
one aspect that needs to be considered.  
  2] In the system, the new words that are seen in test data are completely ignored. We consider
only the words that are present in the training set and the output is generated only based on
the training data that we have. But, in some situations we see that these left out words, do
contribute considerably to the output.  

**Reference:**  
  1] Speech and Language Processing (3rd ed. Draft) Dan Jurafsky and James H. Martin
https://web.stanford.edu/~jurafsky/slp3/  
