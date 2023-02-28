#Courtesy: ibraheemmoosa
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
