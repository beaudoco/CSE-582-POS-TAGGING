import numpy
import math

def index_based_encoding(sentences):
    keys = create_ids(sentences)
    key_list = list(keys)
    max_len = len(max(sentences, key = len))

    idx_based_encoding = []
    for sentence in sentences:
        sentence_encoding = []
        for i in range(max_len):
            if i < len(sentence):
                sentence_encoding.append(key_list.index(sentence[i]) + 1)
            else:
                sentence_encoding.append(0)
        idx_based_encoding.append(sentence_encoding)
    return idx_based_encoding

def bag_of_words_encoding(sentences):
    # NEED TO ADJUST TO ACCOUNT FOR SENTENCES 
    keys = create_ids(sentences)
    keys = list(keys)

    bag_vector = []
    # for sentence in sentences:
    # for w in sentence:
    for sentence in sentences:
        sentence_encoding = []
        for word in keys:
            count = sentence.count(word)
            sentence_encoding.append(count)
        bag_vector.append(sentence_encoding)

    return bag_vector

def create_ids(sentences):
    sentence_data = []
    for sentence in sentences:
        for word in sentence:
            sentence_data.append(word)

    return {value for value in sentence_data}

def term_freq_dict(sentences):
    tf_dict = {}
    i = 0
    for sentence in sentences:
        sentence_dict = {}
        for word in sentence:
            if word not in sentence_dict.keys():
                sentence_dict[word] = sentence.count(word)
        tf_dict[i] = sentence_dict
        i += 1
    return tf_dict

def calculate_tf(word, sentence_num, tf_dict):
    sentence_dict = tf_dict[int(sentence_num)]
    return sentence_dict[word] / sum(sentence_dict.values())

def calculate_idf(word, tf_dict, keys):
    doc_num = 0
    for _, val in tf_dict.items():
        if word in val.keys():
            doc_num += 1
    return math.log(len(keys) / (doc_num+1)) 

def tf_idf(word, sentence_num, tf_dict, keys):
    return round(calculate_tf(word, sentence_num, tf_dict) * calculate_idf(word, tf_dict, keys), 5)

def tf_idf_encoding(sentences):
    tf_idf_encoding = []
    keys = create_ids(sentences)
    tf_dict = term_freq_dict(sentences)

    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence_encoding = []
        for word in keys:
            if word in sentence:
                sentence_encoding.append(tf_idf(word, i, tf_dict, keys))
            else:
                sentence_encoding.append(0)
    
        tf_idf_encoding.append(sentence_encoding)
    return tf_idf_encoding