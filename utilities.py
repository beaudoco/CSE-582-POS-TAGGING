from collections import Counter


# TODO: Handle test data file.
def read_data(fpath, is_training=True):
    """Read the data file and returns the sentences (words and pos tags)

    Parameters:
    fpath (str): Data file path.
    is_training (bool): Whether the file contains training data.

    """
    data = []
    with open(fpath) as f:
        current_sentence = []
        current_pos_tags = []
        for line in f:
            # check if it is an empty new line
            # if it is then this is the end of the current sentence.
            if len(line.strip()) == 0:
                data.append((current_sentence.copy(), current_pos_tags.copy()))
                current_sentence.clear()
                current_pos_tags.clear()
                continue
            word, pos_tag, _ = line.split()
            # Strip whitespaces just in case.
            word = word.strip()
            pos_tag = pos_tag.strip()
            current_sentence.append(word)
            current_pos_tags.append(pos_tag)
    return data

def create_id_mapping_from_counter(counter):
    """Create word to integer mapping based on word counts ie most frequent word gets id 0 and so on.
    """
    id_mapping = dict(map(lambda t: (t[1][0], t[0]), enumerate(counter.most_common())))
    return id_mapping


if __name__ == '__main__':
    data = read_data('data/train.txt')
    print('Read {} sentences.'.format(len(data)))

    words = Counter()
    pos_tags = Counter()
    for sentence, tags in data:
        words.update(sentence)
        pos_tags.update(tags)

    print('Unique words: {}, Total words in the corpus: {}.'.format(len(words), words.total()))
    print('Unique POS Tags: {}'.format(len(pos_tags)))

    print('Most common words')
    print(words.most_common(20))
    print('Most common POS tags')
    print(pos_tags.most_common(20))
