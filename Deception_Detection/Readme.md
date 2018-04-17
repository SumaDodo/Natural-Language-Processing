**Understanding:**  

Deception Detection is a text classification problem. It requires the system to be able to detect
the truthful and deceptive reviews based on analysis of the sentiment of the given sentence.
Here, understanding the sentiment is the most important aspect that needs to be considered in
order to classify the data.

**Analysis:**  

Looking into the training set for deceptive and truthful data gave many insights about the kind
of language, context that is used in both. These were certain features/sentiments observed in
the data which is incorporated in the program:  
  1] The truthful reviews tend to use more nouns than the deceptive ones.  
  2] The use of verb, adverbs and adjectives was seen more in the deceptive reviews.  
  3] Also, one thing observed was the fact that truthful reviews had comparatively more usage of
the punctuation marks than in the deceptive ones.  
  4] The truthful reviews had more of hotel related words describing the place where they stayed
like: adjoining rooms, amenities, baggage, breakfast, check in, check out, reservation etc.
These four set of features were added to the naive bayes implementation.  

**Implementation:**  

  1. Removal of stop words: This was done using the NLTK stopword corpus.  
  2. Lemmatization of the words: NLTK WordNetLemmatizer was used for this.  
  3. NLTK pos tagger was used to get the tag of each word  
  4. The above specified features were considered i.e.,  
      1. The presence of nouns  
      2. Verbs, adverbs and adjectives  
      3. Punctuation marks  
      4. Hotel related words  
These were obtained for both the truthful and deceptive training set and were fed to the naive
bayes classifier in addition to the original inputs that it takes into account like the word
probability of occuring in the truthful reviews and deceptive ones.  
  5. For the test data, the probability of each word in the sentence belonging to a particular class
is checked from the training and is multiplied with the prior probability of the class in the
dataset and along with this the presence of nouns, verbs, adverbs, adjectives, punctuation
marks is checked for every sentence and the probability of these is also multiplied.  

**Results and Discussion:**  

  1] The training data was divided into training-dev in the ratio 80:20. The system accuracy
observed was in the range of 50% to 60% for different training-dev sets.  

**Improvements that can be incorporated in the system:**  

  1] The fact that deceptive reviews tend to have more of strong emotional content rather than
just talking about the experience. Phrases like ‘extremely disappointed’ , ‘extremely
comfortable’ are seen more in the deceptive reviews. This was not considered.  
  2] The system was developed only on the unigrams. Considering bi-grams and tri-grams to
check for certain phrases would have helped.  

**References:**  

  1] Speech and Language Processing (3rd ed. Draft) Dan Jurafsky and James H. Martin
https://web.stanford.edu/~jurafsky/slp3/ [Text Classification]  
  2] Jiwei Li et al. Towards a General Rule for Identifying Deceptive Opinion Spam  
