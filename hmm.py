from collections import defaultdict
from hmmlearn.hmm import CategoricalHMM
from sklearn.model_selection import train_test_split
import numpy as np
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

    word_to_id, id_to_word = create_id_mapping_from_counter(unique_words)
    tag_to_id, id_to_tag = create_id_mapping_from_counter(unique_pos_tags)

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
    print('Baseline model accuracy: {} Missing predictions: {}'.format(correct/total, missing / total))

    initial_tag_probs, tag_to_word, tag_to_tag = calculate_hmm_parameters(data)

    num_pos_tags = len(unique_pos_tags)
    vocab_size = len(unique_words)

    initial_tag_probs_array = np.zeros((num_pos_tags,))
    for tag, tag_id in tag_to_id.items():
        initial_tag_probs_array[tag_id] = initial_tag_probs[tag]

    tag_to_word_array = np.zeros((num_pos_tags, vocab_size))
    for tag, tag_id in tag_to_id.items():
        for word, word_id in word_to_id.items():
            tag_to_word_array[tag_id][word_id] = tag_to_word[tag][word]

    tag_to_tag_array = np.zeros((num_pos_tags, num_pos_tags))
    for tag1, tag1_id in tag_to_id.items():
        for tag2, tag2_id in tag_to_id.items():
            tag_to_tag_array[tag1_id][tag2_id] = tag_to_tag[tag1][tag2]


    # Set up the HMM model
    # Note that we set the params and init_params to empty string.
    # This ensures that the model is not trained when we call fit.
    # We have to call fit as otherwise we get some NotFitted exception.
    # It is likely that we are exploiting some undocumented feature of the library to
    # force the model to not fit on data and use the params we provide.
    model = CategoricalHMM(n_components=len(unique_pos_tags), params='', init_params='')
    model.n_features = len(unique_words)
    model.start_prob_ = initial_tag_probs_array
    model.transmat_ = tag_to_tag_array
    model.emissionprob_ = tag_to_word_array

    # Fit the model on dummy data, I am like 99.99 pct sure this does not change the model params.
    sentence = ["I", "met", "you", "at", "the", "blood", "bank"]
    sentence_word_ids = np.array([word_to_id[w] for w in sentence]).reshape((-1, 1))
    model.fit(sentence_word_ids)

    predicted_pos_tag_ids = model.predict(sentence_word_ids)
    predicted_pos_tags = [id_to_tag[tag] for tag in predicted_pos_tag_ids]
    # Seems to get the correct POS tags
    print(sentence)
    print(predicted_pos_tags)

    # Get accuracy on validation split.
    correct, incorrect = 0, 0
    for sentence, pos_tags in test_data:
        sentence_word_ids = np.array([word_to_id[w] for w in sentence]).reshape((-1, 1))
        predicted_pos_tag_ids = model.predict(sentence_word_ids)
        predicted_pos_tags = [id_to_tag[tag] for tag in predicted_pos_tag_ids]
        for predicted_tag, tag in zip(predicted_pos_tags, pos_tags):
            if predicted_tag == tag:
                correct += 1
            else:
                incorrect += 1
    total = correct + incorrect
    print('HMM Accuracy on validation set: {}'.format(correct/total))


