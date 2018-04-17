**Understanding:**  
Named Entity Recognition is a sequence labelling task. Here we try finding the required entities
by naming them with the tags IOB.

**Implementation:**  
  1] The implementation is based on HMM method. There were many variations that were considered.  
  2] First applied the bigram model with unknown word handling. The system F1 measure was
quite low as there was no unknown word handling and also the correlation among the words
was not considered.  
  3] The system was modified to implement a trigram model. Which improved the performance.
Further, regular expressions were used to handle the unknown words. The shape of the
unknown words is captured through regular expression and is compared with the similar shape
of the known word and similar weight was assigned to the unknown word.  
  4] Further, instead of count, the word embedding of the words was considered. Word2Vec is
used to get the word embedding and this was considered in place of count of the words. This
improved the performance slightly.  

**Results and Discussion:**
In the initial phase, the training was divided 90-10. It was trained on 90% of data and tested on
the rest.  
  1] The simple HMM bigram model  
     1) For unknown words considering the maximum value in viterbi implementation  

||Precision|Recall|F1|
|---|:---:|:---:|---:|       
|Values| 0.52857 |0.37373| 0.4378698|

  2] For the given fact that for sequence labelling, Named Entity Recognition, The relation among
the words is the most important factor, and since HMM model is based on independence of the
words, the system was modified to Trigram model  

**For trigram model:**  

||Precision|Recall|F1|
|---|:---:|:---:|---:|
|Values|0.464646| 0.464646| 0.464646|

  3] Since features are the most important factors and the context of the word in the given
sentence is very important for this problem, word embeddings was considered instead of the
calculation count.  

||Precision| Recall| F1|
|---|:---:|:---:|---:|
|Values |0.5588 |0.4360515| 0.4898746|

**What HMM failed to recognize?**  
Upon examining the dev data, the following observations were made:
  1] HMM doesn’t take into account the dependency of the words at all.  
  2] The most important part is the feature generation. Apart for the features like the word and
tag count, it’s very hard to imbibe multiple features in HMM.  
  3] For the scenarios where the word has been tagged as I in the training set, HMM tags it as I
instead of B taking into account just the count and not considering context.  

**References:**  
  1] Speech and Language Processing (3rd ed. Draft) Dan Jurafsky and James H. Martin
https://web.stanford.edu/~jurafsky/slp3/  
  2] Neural Network Methods for Natural Language Processing by Yoav Goldberg.  
  3] The Word2Vec tutorials like: https://medium.com/ai-society/jkljlj-7d6e699895c4 and
   https://taylorwhitten.github.io/blog/word2vec  
