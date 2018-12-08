import sys, pickle, os, random
import numpy as np
import pandas as pd
import json
import codecs

# tag2, BIO

tag2label = {
    "O": 0,
    "B_NAME": 1,
    "I_NAME": 2,
    "B_PROD": 3,
    "I_PROD": 4
}

label2tag = {
    0: "O",
    1: "B_NAME",
    2: "I_NAME",
    3: "B_PROD",
    4: "I_PROD"
}


def read_source_data(file_path):
    """
    Read source data, and return back files
    :param file_path:
    :return:
    """
    return pd.read_csv(file_path, sep='\t', header=None)


def build_vocabary(vocab_path, corpus_path, min_count, tag2label):
    """
    save the frequency of the words, remove low frequency words
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :param tag2label:
    :return:
    """
    data = read_source_data(corpus_path)
    word2id = {}
    id2word = {}
    label2tag = {}
    for index, row in data.iterrows():
        word = data.iloc[index, 0]
        # the start character / label
        if word == "start":
            continue
        if word not in word2id:
            word2id[word] = [len(word2id) + 1, 1]
        else:
            word2id[word][1] += 1
    print(len(word2id))
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count:
            low_freq_words.append(word)

    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        id2word[new_id] = word
        new_id += 1

    for tag in tag2label.keys():
        label2tag[tag2label.get(tag)] = tag

    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    id2word[new_id] = '<UNK>'
    id2word[0] = '<PAD>'

    print('the length of the word2id {}'.format(len(word2id)))
    with open(vocab_path, 'wb') as fw:
        pickle.dump([word2id, id2word, tag2label, label2tag], fw)
    return word2id


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def random_embedding(vocab, embedding_dim):
    """
    random embedding the initial words
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_matrix = np.random.uniform(-0.25, 0.25, [len(vocab), embedding_dim])
    embedding_matrix = np.float32(embedding_matrix)
    return embedding_matrix


def padding_sentence(sequences, padding_mark = 0):
    """
    fix the length of the sequences, replace 0 when not long enough
    :param sequence:
    :param padding_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    sequence_list, sequences_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        # fix the length
        seq_ = seq[:max_len] + [padding_mark] * max(max_len - len(seq), 0)
        sequence_list.append(seq_)
        # the real length
        sequences_len_list.append(min(len(seq), max_len))

def init_embedding(embedding_path, size, embedding_dim):
    """

    :param embedding_path:
    :param size:
    :param embedding_dim:
    :return:
    """
    embedding_matrix = np.zeros((size, embedding_dim), dtype=np.float32)
    size = os.stat(embedding_path).st_size()
    with open(embedding_path, 'rb') as f:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(f)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx + chunk_size, :] = chunk
            idx +=chunk_size
            pos = f.tell()
    return embedding_matrix[1:]

def generate_mapping(sentences, labels, vocab):
    sentence_id = []
    label_id = []
    real_length = []

    max_length = max([len(x.split(" ")) for x in sentences])
    word2id = vocab[0]

    for i in range(len(sentences)):
        word_ids = np.zeros(max_length, np.int64)
        label_ids = np.zeros(max_length, np.int64)
        words = sentences[i].split(' ')
        label = labels[i].split(' ')
        real_length.append(min(len(words), len(label)))
        for j in range(max_length):
            if j < min(len(words), len(label)):
                if words[j] not in word2id:
                    word_ids[j] = word2id.get('<UNK>')
                    label_ids[j] = tag2label.get(label[j])
                    continue
                word_ids[j] = word2id.get(words[j])
                label_ids[j] = tag2label.get(label[j])
            else:
                word_ids[j] = word2id.get('<PAD>')
                label_ids[j] = tag2label.get('O')
        sentence_id.append(label_ids)
        label_id.append(label_ids)
    return sentence_id, label_id, real_length


def generate_embeddings(word2id, vocab, embedding_path, embedding_dim = 100):
    embedding_matrix = np.random.uniform(-0.25, 0.25, [len(vocab), embedding_dim])
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) < 100:
                continue
            words = line.split(' ')
            words_ = words[1:]
            word = words[0]
            if word in word2id:
                list = [float(x) for x in words_]
                embedding_matrix[word2id[word]] = list
    return embedding_matrix

