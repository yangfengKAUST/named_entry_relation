import numpy as np
import preprocessing
import random
import pandas as pd


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generate a batch iterator for dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]


def generate_batch(data, batch_size, vocab, tag2label, shuffle=True):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []

    for(sent_, tag_) in data:
        sent_ = preprocessing.sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = []

        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels


def word_embedding(embedding_path, embedding_dim, vocab_size, vocab):
    embeddings= np.random.uniform(-1.0, 1.0, size = [vocab_size, embedding_dim])
    used_words = set()
    embeddings_vocab = {}
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            split_line = line.strip().split(" ")
            if len(split_line) != embedding_dim + 1:
                continue
            word = split_line[0]
            embedding = split_line[1:]
            if word in vocab:
                used_words.add(word)
                embeddings[vocab[word] - 1] = list(map(float, embedding))
                embeddings_vocab[vocab[word] - 1] = [word, embedding]
    return embeddings, embeddings_vocab


def split_data():
    """
    Split origin data into training  / testing
    :return:
    """
    data = pd.read_csv('./data/result.txt', seq='\t', header=None)
    # Split train / test data
    # todo this is crude
    dev_sample_percentage = 0.1
    size = int(data.shape[0] * (1 - dev_sample_percentage)) + 1
    train_data = data[:size]
    test_data = data[size:]
    test_data = test_data.reset_index(drop=True)

    train_sentence = []
    temp_sentence = []
    train_sentence_labels = []
    temp_sentence_label = []
    test_sentence = []
    test_sentence_labels = []
    for index, row in train_data.iterrows():
        if train_data.iloc[index, 0] == "start" and train_data.iloc[index, 1] == "start":
            train_sentence.append(temp_sentence)
            train_sentence_labels.append(temp_sentence_label)
            temp_sentence = []
            temp_sentence_label = []
        else:
            temp_sentence.append(train_data.iloc[index, 0])
            temp_sentence_label.append(train_data.iloc[index, 1])

    temp_sentence = []
    temp_sentence_label = []
    for index, row in test_data.iterrows():
        if test_data.iloc[index, 0] == "start" and test_data.iloc[index, 1] == "start":
            test_sentence.append(temp_sentence)
            test_sentence_labels.append(temp_sentence_label)
            temp_sentence = []
            temp_sentence_label = []
        else:
            temp_sentence.append(test_data.iloc[index, 0])
            temp_sentence_label.append(test_data.iloc[index, 1])

    train_data.to_csv('./data/train.txt', header=None, index=False, sep='\t')
    test_data.to_csv('./data/test.txt', header=None, index=False, sep='\t')

    file_train_data = open('./data/train_sentence.txt', 'w')
    file_train_label = open('./data/train_label.txt', 'w')
    file_test_data = open('./data/test_sentence.txt', 'w')
    file_test_label = open('./data/test_label.txt', 'w')

    for sentence in train_sentence:
        if len(sentence) == 0:
            continue
        else:
            file_train_data.write(' '.join(sentence))
            file_train_data.write('\n')
    for label in train_sentence_labels:
        if len(label) == 0:
            continue
        else:
            file_train_label.write(' '.join(label))
            file_train_label.write('\n')
    for test_sent in test_sentence:
        if len(test_sent) == 0:
            continue
        else:
            file_test_data.write(' '.join(test_sent))
            file_test_data.write('\n')
    for test_label in test_sentence_labels:
        if len(test_label) == 0:
            continue
        else:
            file_test_label.write(' '.join(test_label))
            file_test_label.write('\n')

    print(data.shape[0])
    return train_sentence, train_sentence_labels, test_sentence, test_sentence_labels


def get_train_test_data(sentences, labels, real_length):
    percentage = 0.8 * len(sentences)
    percentage_test = int(0.9 * len(sentences))
    train_content = []
    train_label = []
    test_content = []
    test_label = []
    train_length = []
    test_length = []
    valid_content = []
    valid_label = []
    valid_length = []
    for i in range(len(sentences)):
        if i < percentage:
            train_content.append(sentences[i])
            train_label.append(labels[i])
            train_length.append(real_length[i])
        elif i < percentage_test:
            valid_content.append(sentences[i])
            valid_label.append(labels[i])
            valid_length.append(real_length[i])
        else:
            test_content.append(sentences[i])
            test_label.append(labels[i])
            test_length.append(real_length[i])
    return train_content, train_label, test_content, test_label, train_length, test_length, valid_length, valid_content, valid_label


def transform_data(file_path):
    sentences = []
    labels = []
    temp_sentences = []
    temp_labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip('\n')
        words = line.strip('\t')
        if words[0] == 'start' and words[1] == "start":
            sentences.append(temp_sentences)
            labels.append(temp_labels)
            temp_sentence = []
            temp_labels = []
        else:
            temp_sentence.append(words[0])
            temp_labels.append(words[1])
    file_sentence_data = open('./data/train_sentence.txt', 'w')
    file_label_data = open('./data/train_label.txt', 'w')

    for sentence in sentences:
        if len(sentence) == 0:
            continue
        else:
            file_sentence_data.write(' '.join(sentence))
            file_sentence_data.write('\n')
    for label in labels:
        if len(label) == 0:
            continue
        else:
            file_label_data.write(' '.join(label))
            file_label_data.write('\n')


def trans_label_into_file(label, label2tag, sentence, vocab):
    """

    :param label:
    :param label2tag:
    :param sentence:
    :param vocab:
    :return:
    """
    list_belongs = []
    list_sentence = []
    words = []
    word = []
    list_label = []
    for i in range(len(label)):
        if label[i] > 0:
            list_belongs.append(label2tag.get(label[i]))
            list_sentence.append(sentence[i])
            list_label.append(label[i])
    return list_belongs, list_sentence, words, list_label