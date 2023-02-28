import os
import gensim
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


'''# GENERIC PLACEHOLDER UNTIL OUR MODEL TYPE IS DECIDED
# TODO: UPDATE W/ APPROPRIATE NAME
model_name = "NeuralNets"
name = f"model/NeuralNets/{model_name}"
log_dir = f"{name}"

data_dir = "data/train.txt"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)'''


#Read in words to train
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
        data.append(word[0]) #make lower and experiment
        labels.append(word[1])
        s.append(word[0])

#After testing, show results of predicted WE with actual
def test_predicted_list(words, predicted, actual): #len= 832
    compare_list=[]

    for i in range(len(words)):
        compare_list.append([words[i], predicted[i], actual[i]])


    #Write to text file
    with open(r'./output.txt', 'w') as fp:
        for item in compare_list:
            # write each item on a new line
            fp.write("%s\n" % item)


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
def word_embeddings():
    path = Path('./vectors.kv')

    if path.is_file() is False: #If we need to generate word embeddings
        #Current implementation at 68% for testing
        model = gensim.models.Word2Vec(min_count = 15, negative= 20, window=1, epochs= 15, vector_size=100, sg=0) #Can change parameters
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=15) 

        word_vectors = model.wv
        word_vectors.save('vectors.kv')
        word_dict= word_vectors.key_to_index #all words and their index
    
    else: #Vector file already exists
        word_vectors = KeyedVectors.load('vectors.kv')
        word_dict= word_vectors.key_to_index

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
    ## Specific to Word2Vec Embeddings 
    values= list(data.values())
    words_before= list(data.keys())
    word_embs=[]
    labels=[]
    words=[]
    
    for v in range(len(values)):
        if isinstance(values[v], list):
            word_embs.append(list(values[v][0]))
            labels.append(values[v][1])
            words.append(words_before[v])
    #Until here, can comment out for other types of embeddings

    #Split data
    split = int(len(word_embs) * .15)
    X_train = word_embs[split:]
    y_train = labels[split:]
    X_test = word_embs[:split]
    y_test = labels[:split]
    word_test = words[:split]

    #Build the model
    class_model = LogisticRegression(max_iter=300)
    class_model.fit(X_train, y_train)

    #Testing
    y_pred = class_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    #test_predicted_list(word_test, y_pred, y_test)

word_embed_dict= word_embeddings() #Key= word, Value= [Vector_embeddings, POS-Tag]
Logistic_Regression(word_embed_dict)
