# Import necessary Libraries

from utilities import * # Courtesy: @ibraheemmoosa for read_data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Reading the given Corpus to train and find the tagged sentences
tagged_sentences= read_data("data/train.txt")
print(f'First tagged sentence: {tagged_sentences[0]}')
print(f'# Tagged sentences: {len(tagged_sentences)}')


# Feature Preprocessing
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }
 


# Training-Test Split of Dataset
# Split the dataset for training and testing
training_sentences, test_sentences = train_test_split(tagged_sentences, test_size= 0.3, random_state= 3)
 
print(f'# Training sentences: {len(training_sentences)}')   # 6255
print(f'# Test sentences: {len(test_sentences)}')         # 2681 for 70-30 train-test split
 
def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged[0])):
            
            X.append(features(tagged[0], index))
            y.append(tagged[1][index])
 
    return X, y

X, y = transform_to_dataset(training_sentences) 


# Using MLP Classifier

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', MLPClassifier(hidden_layer_sizes= 20, learning_rate_init= 0.05, random_state=1, max_iter= 5))
])

clf.fit(X[:10000], y[:10000])   # Using only the first 10K samples 
print('Training completed.')

print('Testing on test data...')

X_test, y_test = transform_to_dataset(test_sentences)
preds = clf.predict(X_test)
print(f'Accuracy :{accuracy_score(y_test, preds)}')
