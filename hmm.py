from collections import defaultdict
from hmmlearn.hmm import MultinomialHMM
from sklearn.model_selection import train_test_split
from utilities import *


def calculate_baseline_parameters(data):
    word_to_tag_counter = defaultdict(Counter)

    for sentence, pos_tags in data:
        for word, tag in zip(sentence, pos_tags):
            word_to_tag_counter[word][tag] += 1

    word_to_tag = dict()
    for word in word_to_tag_counter:
        max_count = 0
        max_count_tag = None
        for tag in word_to_tag_counter[word]:
            if word_to_tag_counter[word][tag] > max_count:
                max_count = word_to_tag_counter[word][tag]
                max_count_tag = tag
        word_to_tag[word] = max_count_tag

    return word_to_tag


def calculate_hmm_parameters(data):
    initial_tag_counter = Counter()
    tag_to_word_counter = defaultdict(Counter)
    tag_to_tag_counter = defaultdict(Counter)
    for sentence, pos_tags in data:
        initial_tag_counter[pos_tags[0]] += 1
        for word, tag in zip(sentence, pos_tags):
            tag_to_word_counter[tag][word] += 1
        for i in range(1, len(pos_tags)):
            tag_to_tag_counter[pos_tags[i-1]][pos_tags[i]] += 1
    
    # Calculate initial POS tag probabilities
    initial_tag_probs = defaultdict(float)
    for tag in initial_tag_counter:
        initial_tag_probs[tag] = initial_tag_counter[tag] / initial_tag_counter.total()

    # Calculate tag to tag transition probabilities
    tag_to_tag_probs = defaultdict(lambda: defaultdict(float))
    for tag in tag_to_tag_counter:
        for next_tag in tag_to_tag_counter[tag]:
            tag_to_tag_probs[tag][next_tag] = tag_to_tag_counter[tag][next_tag] / tag_to_tag_counter[tag].total()

    # Calculate tag to word emission probabilities
    tag_to_word_probs = defaultdict(lambda: defaultdict(float))
    for tag in tag_to_word_counter:
        for word in tag_to_word_counter[tag]:
            tag_to_word_probs[tag][word] = tag_to_word_counter[tag][word] / tag_to_word_counter[tag].total()

    return initial_tag_probs, tag_to_word_probs, tag_to_tag_probs


if __name__ == '__main__':
    data = read_data('data/train.txt')
    unique_words = Counter()
    unique_pos_tags = Counter()

    for sentence, pos_tags in data:
        unique_words.update(sentence)
        unique_pos_tags.update(pos_tags)

    word_to_id = create_id_mapping_from_counter(unique_words)
    tag_to_id = create_id_mapping_from_counter(unique_pos_tags)

    train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)


    word_to_tag = calculate_baseline_parameters(train_data)
    correct, incorrect, missing = 0, 0, 0
    for sentence, pos_tags in test_data:
        for word, tag in zip(sentence, pos_tags):
            if word in word_to_tag:
                if word_to_tag[word] == tag:
                    correct += 1
                else:
                    incorrect += 1
            else:
                missing += 1
    total = correct + incorrect + missing
    print(correct/total)
    print(missing / total)

    # initial_tag, tag_to_word, tag_to_tag = calculate_hmm_parameters(data)
    # print(initial_tag)
    # print(tag_to_tag)
