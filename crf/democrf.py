# TODO add parser maybe polyglot. but performance seems reasonable.
# need more extensive measure of performance. Consider cross validation
# Also tags need \n stripping

import pycrfsuite
from pathlib import Path

target = Path("c:/","users","rhiggins","projects","data_prep", "DF_LOC_tagged_data.txt")
with open(target,'r', encoding='utf-8') as f:
    total_sent = []
    sent = []
    for line in f:
        token,tag = line.split('\t')
        sent.append((token,tag))
        if token=='.':
            total_sent.append(sent)
            sent=[]

fair_sent = [sent for sent in total_sent if len(sent)>2 ]


def word2features(sent, i):
    word = sent[i][0]
    # postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        # 'postag=' + postag,
        # 'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        # postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            # '-1:postag=' + postag1,
            # '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            # '+1:postag=' + postag1,
            # '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


from sklearn.model_selection import  train_test_split

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for token, label in doc]

data = fair_sent
X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

import numpy as np
from sklearn.metrics import classification_report

labels = {'O-LOC\n':0,'B-LOC\n':1,'I-LOC\n':1}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

# Print out the classification report
print(classification_report(
    truths, predictions,))