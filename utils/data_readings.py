def sentence_structured_data(data_dir):
    file = open(data_dir, "r")
    sentence_labels = []
    labels = []
    sentences = []
    sentence = ''
    for line in file:
        word = line.split(' ')    
    
        if word[0] == '\n':
            sentences.append(sentence[:-1])
            sentence = ''

            labels.append(sentence_labels)
            sentence_labels = []
        else:
            sentence += word[0].lower() + ' '
            sentence_labels.append(word[1])
    
    return sentences, labels