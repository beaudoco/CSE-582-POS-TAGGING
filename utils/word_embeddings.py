import numpy

def index_based_encoding(data):
    keys = {}
    keys = {value for value in data}

    key_list = list(keys)
    data_ids = []

    for key in data:
    # print('key: {}, index: {}'.format(key, key_list.index(key) + 1))
        data_ids.append(key_list.index(key) + 1)
    return data_ids

def bag_of_words_encoding(data):
    # NEED TO ADJUST TO ACCOUNT FOR SENTENCES 
    words = data
    keys = {value for value in data}

    bag_vector = numpy.zeros(len(keys))
    # for sentence in sentences:
    # for w in sentence:
    for w in words:
        for i,word in enumerate(keys):
            if word == w:
                bag_vector[i] += 1
    return bag_vector