**Problem:**  
POS-tagged data corpus was used as training dat. The sentences in the training set are arranged as one word per line with a blank line 
separating the sentences. The columns are tab separated, with the first column giving word position,
the second the word and the third the POS tag.

Implementation was done using Viterbi decoder. The results are discussed further.

**Understanding:**  
The POS tag system designed is a supervised learning problem where the system is trained
with the set of training data and is expected to learn from it and predict the output for the test
data. Since we are supposed to find the most appropriate sequence of the hidden states [here
tags] for the given set of words, we use Viterbi Algorithm.

**Design:**  
Initially the training data was split into training and test data with 90:10 ratios and the system
was designed based on this training set and was tested upon the corresponding test data. The
basic understanding is that the system must be able to deal with the unknown words. The
system might not be trained accurately for this as the ratio of the split up 90:10 which means
there might not be any unknown words seen in test data.
Prior to the actually implementation, we need:  
  1) Bigram Count of tag,tag and word given tag  
  2) Transition Probability matrix: Probability of transition from one state (tag) to another.  
  3) Emission Probability matrix: Probability of the occurrence word given tag.  
  4) Start Probability: Since we are not given as to what word will occur at the beginning,
we need to have the probability of the tags occurring immediately after the start state.  

**Implementation:**
  1) Choice of the matrix type: Initially tried to use dictionary and list to handle all the
matrices but it was way too complex when it came to dividing the elements of the
transition matrix and the emission matrix. Thus, I chose to implement it with numpy.  
  2) How even the declaration of the type of the numpy array played its role: The matrix
seemed to hold the redundant values with data type as just float as the division
operation often exceeded the size in float. Thus one aspect that played its role was
declaration of type as float64. The probabilities stored in matrix tend to be with the
more precision value than the ones when previously declared as just float.  
  3) Transition matrix smoothing: Even though I tried discounting for smoothing, what
worked out well for this model was the Laplace Smoothing.  
  4) Start Probability matrix: It is important to recognize the start and end of the sentence
and to include start and end in probability computation. For this system, all the tags
are given the equal probability.  
  5) Unknown Words: Since there are high chances of unknown words being the names of
the place or name of the person, So, the unknown words are given the probability of
the tag which had the highest probability value in training data [which in most cases is
noun]  

**Output and Results:**  
The accuracy for the 90:10 training vs. dev split up observed was 94.97% compared to the
88% of the baseline model. I believe this difference is due to the fact that a word can act as
noun and verb as well but in the baseline, if I happen to see that word, I blindly assign it the
tag that had the high probability in the training set. This was curbed to some extent through
the Viterbi algorithm implementation.

However, on the test set, acurracy observed was 91.5%

**References:**
  1) Speech and Language Processing ( 3rd ed. Draft) Dan Jurafsky and James H. Martin
https://web.stanford.edu/~jurafsky/slp3/  
  2) https://en.wikipedia.org/wiki/Viterbi_algorithm  
  3) Numpy SciPy.org Quick Start tutorial.
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html  
