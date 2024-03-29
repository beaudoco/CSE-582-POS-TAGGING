from collections import Counter, defaultdict


# TODO: Handle test data file.
def read_data(fpath, is_training=True, lowercase=False):
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
            if lowercase:
                line = line.lower()
            # check if it is an empty new line
            # if it is then this is the end of the current sentence.
            if len(line.strip()) == 0:
                if is_training:
                    data.append((current_sentence.copy(), current_pos_tags.copy()))
                else:
                    data.append(current_sentence.copy())
                current_sentence.clear()
                if is_training:
                    current_pos_tags.clear()
                continue
            if is_training:
                word, pos_tag, _ = line.split()
            else:
                word = line

            # Strip whitespaces just in case.
            word = word.strip()
            if is_training:
                pos_tag = pos_tag.strip()
            current_sentence.append(word)
            if is_training:
                current_pos_tags.append(pos_tag)
    return data

def create_id_mapping_from_counter(counter, add_unk=False):
    """Create word to integer mapping based on word counts ie most frequent word gets id 0 and so on.
    """
    if add_unk:
        id_mapping = defaultdict(int)
    else:
        id_mapping = dict()
    id_reverse_mapping = dict()
    if add_unk:
        id_reverse_mapping[0] = 'UNK'
        idx = 1
    else:
        idx = 0
    for item, count in counter.most_common():
        id_mapping[item] = idx
        id_reverse_mapping[idx] = item
        idx += 1
    return id_mapping, id_reverse_mapping


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
