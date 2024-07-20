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

print(Kmers_funct(mySeq, size=7))

words = Kmers_funct(mySeq, size = 6)
joined_sentence = ' '.join(words)
sentence1 = ' '.join(Kmers_funct(mySeq1, size = 6))
sentence2 = ' '.join(Kmers_funct(mySeq2, size = 6))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform([joined_sentence, sentence1, sentence2]).toarray()
print(X)

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    break

human_dna = pd.read_table('input/human.txt')
print(human_dna.head())
human_dna['class'].value_counts().sort_index().plot.bar()
plt.title("Class Distribution of Human DNA")
plt.show()

chimp_dna = pd.read_table('input/chimpanzee.txt')
chimp_dna.head()
chimp_dna['class'].value_counts().sort_index().plot.bar()
plt.title("Class distribution of chimpanzee dna")
plt.show()

dog_dna = pd.read_table('input/dog.txt')
dog_dna.head()
dog_dna['class'].value_counts().sort_index().plot.bar()
plt.title("Class distribution of dog dna")
plt.show()

human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence'], size=6), axis=1 )
human_dna = human_dna.drop('sequence', axis = 1)

chimp_dna['words'] = chimp_dna.apply(lambda x: Kmers_funct(x['sequence'], size=6), axis=1)
chimp_dna = chimp_dna.drop('sequence', axis = 1)

dog_dna['words'] = dog_dna.apply(lambda x: Kmers_funct(x['sequence'], size=6), axis=1)
dog_dna = dog_dna.drop('sequence', axis = 1)

print(human_dna.head())

human_texts =



