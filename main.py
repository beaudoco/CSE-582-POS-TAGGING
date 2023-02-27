import os
from utils.data_readings import sentence_structured_data
from utils.word_embeddings import index_based_encoding, bag_of_words_encoding, tf_idf_encoding

print("Hello World")

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

'''Things to do:
    1. Split data into training and testing 
    2. Create word embeddings using gensim Word2Vec (do we need to normalize beforehand? Ex. make all lowercase?)
    3. Hidden Markov Model (Library: https://hmmlearn.readthedocs.io/en/latest/tutorial.html)
    4. Logistic Regression (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    5. Multi-Layer Perceptron (https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
    6. Testing, evaluation metrics, and improving accuracy
'''

sentences, labels = sentence_structured_data(data_dir)
# IDX BASED ENCODING
# idx_encoded_data = index_based_encoding(sentences)
# print(idx_encoded_data)

# BAG OF WORD BASED ENCODING
# bag_vector = bag_of_words_encoding(data[:10])
# print(bag_vector)

# TF IDF BASED ENCODING
# tf_idf_encoding = tf_idf_encoding(sentences)
# print(tf_idf_encoding[0])
