import Bio
import numpy as np
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

from Bio import SeqIO
import sklearn
from sklearn.preprocessing import LabelEncoder

"""for sequence in SeqIO.parse("example_dna.fa", "fasta"):
   print(sequence.id)
    print(sequence.seq)
    print(len(sequence))
    break
"""

def string_to_array(seq_string):
    seq_string = seq_string.lower()
    seq_string = re.sub('[^actg]', 'n', seq_string)
    seq_string = np.array(list(seq_string))
    return seq_string

#label encoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','t','g','z']))

#Ordinal Encoder
'''def ordinal_encoder (my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25
    float_encoded[float_encoded == 1] = 0.5
    float_encoded[float_encoded == 2] = 0.75
    float_encoded[float_encoded == 3] = 1
    float_encoded[float_encoded == 4] = 0.00
    return float_encoded
seq_test= 'TTCAGCCAGTG'

print(ordinal_encoder(string_to_array(seq_test)))
'''

'''
#One Hot encoding
from sklearn.preprocessing import OneHotEncoder
def one_hot_encoder(seq_string):
    int_encoded = label_encoder.transform(seq_string)
    onehot_encoder = OneHotEncoder(sparse_output=False, dtype=int)
    int_encoded = int_encoded.reshape(len(int_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    onehot_encoded = np.delete(onehot_encoded,-1,1)
    return onehot_encoded

seq_test= 'GAATTCTCGAA'
print(one_hot_encoder(string_to_array(seq_test)))
'''

#K-Mer counting

#splits sentence
def Kmers_funct(seq, size):
    return [seq[x:x+size].lower() for x in range(len(seq) - size+1)]
mySeq = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'
mySeq1 = 'TCTCACACATGTGCCAATCACTGTCACCC'
mySeq2 = 'GTGCCCAGGTTCAGTGAGTGACACAGGCAG'

#print(Kmers_funct(mySeq, size=7))

words = Kmers_funct(mySeq, size = 6)
joined_sentence = ' '.join(words)
sentence1 = ' '.join(Kmers_funct(mySeq1, size = 6))
sentence2 = ' '.join(Kmers_funct(mySeq2, size = 6))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform([joined_sentence, sentence1, sentence2]).toarray()
#print(X)

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    break

human_dna = pd.read_table('input/human.txt')
print(human_dna.head())
human_dna['class'].value_counts().sort_index().plot.bar()
plt.title("Class Distribution of Human DNA")
#plt.show()

chimp_dna = pd.read_table('input/chimpanzee.txt')
chimp_dna.head()
chimp_dna['class'].value_counts().sort_index().plot.bar()
plt.title("Class distribution of chimpanzee dna")
#plt.show()

dog_dna = pd.read_table('input/dog.txt')
dog_dna.head()
dog_dna['class'].value_counts().sort_index().plot.bar()
plt.title("Class distribution of dog dna")
#plt.show()

human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence'], size=6), axis=1 )
human_dna = human_dna.drop('sequence', axis = 1)

chimp_dna['words'] = chimp_dna.apply(lambda x: Kmers_funct(x['sequence'], size=6), axis=1)
chimp_dna = chimp_dna.drop('sequence', axis = 1)

dog_dna['words'] = dog_dna.apply(lambda x: Kmers_funct(x['sequence'], size=6), axis=1)
dog_dna = dog_dna.drop('sequence', axis = 1)

#print(human_dna.head())

human_texts = list(human_dna['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])

y_human = human_dna.iloc[:,0].values


chimp_texts = list(chimp_dna['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])

y_chimp = chimp_dna.iloc[:,0].values

dog_texts = list(dog_dna['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])


y_dog = dog_dna.iloc[:,0].values

print(y_human)
print(y_chimp)
print(y_dog)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4, 4))
X = cv.fit_transform(human_texts)
X_chimp = cv.fit_transform(chimp_texts)
X_dog = cv.fit_transform(dog_texts)

print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_human,
                                                    test_size= 0.20,
                                                    random_state=42)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha = 0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix for predictions on chimp \n")
print(pd.crosstab(pd.Series(y_test, name = "actual"), pd.Series(y_pred, name="predicted")))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average = "weighted")
    recall = recall_score(y_test, y_predicted, average = "weighted")
    f1 = f1_score(y_test, y_predicted, average="weighted")
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print(y_pred)
print(y_test)


print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

y_pred_dog = classifier.predict(X_dog)

print(y_pred_dog)
print(y_dog)

print("Confusion matrix for predictions on Dog test DNA sequence\n")
print(pd.crosstab(pd.Series(y_dog, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

