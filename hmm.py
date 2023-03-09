from collections import defaultdict
from hmmlearn.hmm import CategoricalHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
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


def calculate_hmm_parameters(data, tag_to_id, word_to_id, alpha_it=1, alpha_t2t=1, alpha_t2w=1):
    num_of_tags = len(tag_to_id)
    num_of_words = len(word_to_id)

    initial_tag_counter = Counter()
    tag_to_word_counter = defaultdict(Counter)
    tag_to_tag_counter = defaultdict(Counter)
    for sentence, pos_tags in data:
        initial_tag_counter[tag_to_id[pos_tags[0]]] += 1
        for word, tag in zip(sentence, pos_tags):
            tag_to_word_counter[tag_to_id[tag]][word_to_id[word]] += 1
        for i in range(1, len(pos_tags)):
            tag_to_tag_counter[tag_to_id[pos_tags[i-1]]][tag_to_id[pos_tags[i]]] += 1
    
    # Calculate initial POS tag probabilities
    initial_tag_probs = np.zeros((num_of_tags,))
    for tag in range(num_of_tags):
        initial_tag_probs[tag] = (alpha_it + initial_tag_counter[tag]) / (initial_tag_counter.total() + alpha_it * num_of_tags)

    # Calculate tag to tag transition probabilities
    tag_to_tag_probs = np.zeros((num_of_tags, num_of_tags))
    for tag in range(num_of_tags):
        for next_tag in range(num_of_tags):
            tag_to_tag_probs[tag][next_tag] = (alpha_t2t + tag_to_tag_counter[tag][next_tag]) / (tag_to_tag_counter[tag].total() + alpha_t2t * num_of_tags)

    # Calculate tag to word emission probabilities
    tag_to_word_probs = np.zeros((num_of_tags, num_of_words))
    for tag in range(num_of_tags):
        for word in range(num_of_words):
            tag_to_word_probs[tag][word] = (tag_to_word_counter[tag][word] + alpha_t2w) / (tag_to_word_counter[tag].total() + alpha_t2w * num_of_words)

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

    for i in range(len(unique_pos_tags)):
        tag = id_to_tag[i]
        print('Tag: {} P: {}'.format(tag, unique_pos_tags[tag] / unique_pos_tags.total()))


    train_data, test_data = train_test_split(data, test_size=0.1, random_state=41)


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

    train_words = Counter()
    train_pos_tags = Counter()

    for sentence, pos_tags in train_data:
        train_words.update(sentence)
        train_pos_tags.update(pos_tags)

    word_to_id, id_to_word = create_id_mapping_from_counter(unique_words)
    tag_to_id, id_to_tag = create_id_mapping_from_counter(unique_pos_tags)

    initial_tag_probs_array, tag_to_word_array, tag_to_tag_array = calculate_hmm_parameters(train_data, tag_to_id, word_to_id, alpha_it=1.0, alpha_t2t=1.0, alpha_t2w=1e-2)

    num_pos_tags = len(unique_pos_tags)
    vocab_size = len(unique_words)
    

    # Set up the HMM model
    # Note that we set the params and init_params to empty string.
    # This ensures that the model is not trained when we call fit.
    # We have to call fit as otherwise we get some NotFitted exception.
    # It is likely that we are exploiting some undocumented feature of the library to
    # force the model to not fit on data and use the params we provide.
    model = CategoricalHMM(n_components=len(unique_pos_tags), params='', init_params='', random_state=42)
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
    correct_per_tag, incorrect_per_tag = defaultdict(int), defaultdict(int)
    y_true, y_pred = [], []
    for sentence, pos_tags in test_data:
        sentence_word_ids = np.array([word_to_id[w] for w in sentence]).reshape((-1, 1))
        log_prob, predicted_pos_tag_ids = model.decode(sentence_word_ids, algorithm='viterbi')
        # print(log_prob)
        predicted_pos_tags = [id_to_tag[tag] for tag in predicted_pos_tag_ids]
        for predicted_tag, tag in zip(predicted_pos_tags, pos_tags):
            y_true.append(tag_to_id[tag])
            y_pred.append(tag_to_id[predicted_tag])
            if predicted_tag == tag:
                correct += 1
                correct_per_tag[tag] += 1
            else:
                incorrect += 1
                incorrect_per_tag[tag] += 1
    total = correct + incorrect
    print('HMM Accuracy on validation set: {}'.format(correct/total))

    conf_mat = confusion_matrix(y_true, y_pred, normalize='true', labels=list(range(num_pos_tags)))
    tag_names = [id_to_tag[i] for i in range(len(unique_pos_tags))]
    conf_mat = pd.DataFrame(conf_mat, index=tag_names, columns=tag_names)
    sns.heatmap(conf_mat, xticklabels=True, yticklabels=True)
    # disp = ConfusionMatrixDisplay(conf_mat, display_labels=[id_to_tag[i] for i in range(len(unique_pos_tags))])
    # disp.plot()
    plt.show()


    for i in range(len(unique_pos_tags)):
        tag = id_to_tag[i]
        accuracy = (1. + correct_per_tag[tag]) / (2. + correct_per_tag[tag] + incorrect_per_tag[tag])
        print('Tag: {} Acc: {}'.format(tag, accuracy))



