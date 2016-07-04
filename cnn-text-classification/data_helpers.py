#!/usr/bin/python
import numpy as np
import re
import itertools
from collections import Counter
import json
from pprint import pprint
import os



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with open('./data/authors.json') as data_file:
        data = json.load(data_file)

    # pprint(data)

    list_examples = list()
    labels_lenght = len(data)

    list_labels = []
    for d in data:
        index = d['index']
        temp_label = []
        i = 0
        while i < labels_lenght:  # inzializzi la label temporanea con tutti 0
            if i == index:
                temp_label.append(1)
            else:
                temp_label.append(0)
            i += 1
        # temp_label[d['index']] = 1
        temp_examples = list(open('./data/input/' + d['file_name']).readlines())
        temp_examples = [s.strip() for s in temp_examples]
        for _ in temp_examples:
            list_labels.append(temp_label)
        list_examples.append(temp_examples)

    x_text = []
    for exa in list_examples:
        x_text = x_text + exa
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

   # y = np.concatenate(list_labels, 0)
    return [x_text, list_labels]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    voc_writer = open('voc.txt', 'w')
    inv_voc_w = open('inv_voc.txt','w')
    for k in vocabulary.keys():
        voc_writer.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    for i in vocabulary_inv:
        inv_voc_w.write(str(i) + '\n')
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    print('Labels: ' + str(len(labels)))
    print('Words: ' + str(len(labels)))

    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    print('Load data and lables')
    sentences, labels = load_data_and_labels()
    print('Pad sentences')
    sentences_padded = pad_sentences(sentences)
    print('Build vocabularies')
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    print('Build input data')
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
