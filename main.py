import os
from utils.data_readings import sentence_structured_data
from utils.word_embeddings import index_based_encoding, bag_of_words_encoding, tf_idf_encoding
import gensim
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

'''

# GENERIC PLACEHOLDER UNTIL OUR MODEL TYPE IS DECIDED
# TODO: UPDATE W/ APPROPRIATE NAME
model_name = "NeuralNets"
name = f"model/NeuralNets/{model_name}"
log_dir = f"{name}"

data_dir = "data/train.txt"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# READ DATA FROM FILE INTO INPUT & LABELS
# TODO: DECIDE HOW WE WILL PROCESS NEW LINES
file = open(data_dir, "r")
data = []
labels = []
for line in file:
    word = line.split(' ')
    
    if word[0] == '\n':
        # print("NEW LINE")
        continue
    else:
        data.append(word[0].lower())
        labels.append(word[1])
'''
'''Things to do:
    1. Split data into training and testing 
    2. Create word embeddings using gensim Word2Vec (do we need to normalize beforehand? Ex. make all lowercase?)
    3. Hidden Markov Model (Library: https://hmmlearn.readthedocs.io/en/latest/tutorial.html)
    4. Logistic Regression (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    5. Multi-Layer Perceptron (https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
    6. Testing, evaluation metrics, and improving accuracy
'''

# READ DATA FROM FILE INTO INPUT & LABELS
# TODO: DECIDE HOW WE WILL PROCESS NEW LINES
file = open("data/train.txt", "r")
data = []
labels = []
sentences= []
s= []

#Extract words and corresponding labels into lists
#May need some structure to identify sentences
for line in file:
    word = line.split(' ')
    
    if word[0] == '\n':
        sentences.append(s)
        s=[]
    else:
        data.append(word[0].lower())
        labels.append(word[1])
        s.append(word[0].lower())

#Create word embeddings using Word2Vec
#Do we need normalization?? Lower case of words?
'''
    Word2Vec:
        -min_count: Ignores all words with frequency lower than count
        -window: number of words on each side
        -vector_size: dimensionality of feature vector
        -sg: training alg, skipgram=1, CBOW=0
        -negative: if neg sampling used, how many noise words drawn
        -epochs: how many times iterates through training set
    Source: https://radimrehurek.com/gensim/models/word2vec.html 
'''
def word_embeddings(sentences):
    model = gensim.models.Word2Vec(min_count = 5, window=3, vector_size=20, sg=0) #Can change parameters
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10) 

    word_vectors = model.wv
    word_dict= word_vectors.key_to_index #all words and their index

    for keys in word_dict:
        if keys.isalnum(): #disregard punctuation
            vector_POS = []
            vector_POS.append(word_vectors[keys])
            POS_index= data.index(keys)
            vector_POS.append(labels[POS_index])
            word_dict[keys]= vector_POS 

    #word_vectors[word]: gives vector for word

    return word_dict

def Logistic_Regression(data):
    #Separate data
    values= list(data.values())
    word_embs=[]
    labels=[]
    
    for v in values:
        if isinstance(v, list):
            word_embs.append(list(v[0]))
            labels.append(v[1])

    #Build the model
    class_model = LogisticRegression()
    class_model.fit(word_embs, labels)

    #Testing example on word "Pound"
    x_test= data["pound"][0]
    y_test= data["pound"][1]
    y_pred = class_model.predict(x_test.reshape(1,-1))

word_embed_dict= word_embeddings(sentences) #Key= word, Value= [Vector_embeddings, POS-Tag]
#X_train, X_test = train_test_split(word_embed, test_size=0.2)
Logistic_Regression(word_embed_dict)

# IDX BASED ENCODING
# idx_encoded_data = index_based_encoding(sentences)
# print(idx_encoded_data[0])
# print(sentences[0])

# BAG OF WORD BASED ENCODING
# bag_vector = bag_of_words_encoding(sentences)
# print(bag_vector[0])
# print(sentences[0])

# TF IDF BASED ENCODING
# tf_idf_encoding = tf_idf_encoding(sentences[:10])
# print(tf_idf_encoding[0])
# print(sentences[0])