import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import crf_decode

class BILSTM_CRF(object):
    def __init__(self, batch_size,
                 hidden_dim, embeddings, CRF,
                 dropout, num_tags, fine_tune=True):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.CRF = CRF
        self.dropout_keep_prob = dropout
        self.num_tags = num_tags
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None,None], name="labels")
        self.sequence_length = tf.placeholder(tf.int32, shape=[None, ], name="sequence_length")
        self.fine_tune = fine_tune

        # look-up layer
        with tf.name_scope("embedding"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.fine_tune,
                                           name="_word_embeddings")
            self.word_embeddings = tf.nn.embedding_lookup(
                params=_word_embeddings,
                ids=self.word_ids,
                name="word_embeddings"
            )
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout_keep_prob)

        # LSTM
        with tf.name_scope("bi-lstm"):
            # forward LSTM layer
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            # backward LSTM layer
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)

            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_length,
                dtype=tf.float32
            )
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, keep_prob=self.dropout_keep_prob)

        # LSTM output layer
        with tf.name_scope("projection"):
            W = tf.get_variable(name='W',
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            # output shape is [batch_size, steps, cell_num]
            s = tf.shape(output)
            # reshape
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])

            self.pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(self.pred, [-1, s[1], self.num_tags])

        # CRF layer
        with tf.name_scope("loss"):
            if self.CRF:
                self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                                 tag_indices=self.labels,
                                                                                 sequence_lengths=self.sequence_length)
                self.loss = -tf.reduce_mean(self.log_likelihood)

                viterbi_sequence, viteribi_score = crf_decode(potentials=self.logits,
                                                              transition_params=self.transition_params,
                                                              sequence_length=self.sequence_length)
                self.viterbi_sequence = tf.subtract(tf.add(viterbi_sequence, viterbi_sequence),
                                                    viterbi_sequence, name="viterbi_sequence")
                self.viterbi_score = tf.subtract(tf.add(viteribi_score, viteribi_score),
                                                 viteribi_score, name="viterbi_score")
